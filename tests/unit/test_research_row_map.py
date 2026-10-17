"""Research row map, monthly decision rows and split manifest (spec 5.1, 6.1, 11.1; INV-DATA-001): calendars only, no prices."""
from __future__ import annotations

from pathlib import Path

import pytest

from quant_pipeline.calendars import (
    Calendars,
    build_row_map,
    holdout_rows,
    outer_fold_count,
    outer_leftover_rows,
    research_warmup_rows,
)
from quant_pipeline.config import load_config
from quant_pipeline.splits import build_split_manifest

ROOT = Path(__file__).resolve().parents[2]
def _hydrated():
    """The research profile with 10 explicit universe records, as `snapshot.hydrate_universe` would supply from a verified snapshot."""
    from quant_pipeline.config import Asset

    cfg = load_config(ROOT / "configs" / "research_8_12asset.yaml")
    recs = []
    for i, ex in enumerate(["XKRX"] * 5 + ["XNYS"] * 5):
        krx = ex == "XKRX"
        recs.append(Asset(asset_id=f"R{i:02d}", exchange=ex, currency="KRW" if krx else "USD", timezone="Asia/Seoul" if krx else "America/New_York",
                          calendar=ex, raw_price_field="close", model_price_field="total_return_index", lot_size=1, quantity_precision=0,
                          allow_short=False, benchmark_eligible=True, max_weight=0.30))
    return cfg.model_copy(update={"universe": cfg.universe.model_copy(update={"assets": recs})})


CFG = _hydrated()
FIX = load_config(ROOT / "configs" / "fixture_3asset.yaml")
CALS = Calendars.load(["XKRX", "XNYS"])
WINDOW = ("2016-01-04", "2023-12-29")


@pytest.fixture(scope="module")
def maps():
    rm, sess = build_row_map(CFG, CALS, WINDOW)
    return rm, sess, build_split_manifest(CFG, CALS, rm)


def test_warmup_is_the_largest_look_back_and_the_holdout_follows_the_declared_fraction():
    assert research_warmup_rows(CFG) == 252
    assert holdout_rows(CFG, 1603) == 241  # ceil(0.15 * 1603)
    assert holdout_rows(CFG, 300) == 126  # the declared minimum applies to short samples
    assert holdout_rows(FIX, 70) == 10  # tail_rows is unchanged


def test_eligible_rows_are_the_post_warmup_common_sessions(maps):
    rm, sess, _ = maps
    both = sess["XNYS"].intersection(sess["XKRX"])
    assert rm.complete_case == tuple(both[1:]) and rm.warmup == rm.complete_case[:252] and rm.eligible == rm.complete_case[252:]
    assert rm.position(rm.eligible[0]) == 1 and rm.position(rm.warmup[0]) == 0


def test_decisions_are_the_first_eligible_xnys_cutoff_of_each_calendar_month(maps):
    rm, sess, _ = maps
    ds = rm.decisions
    months = [d.session.strftime("%Y%m") for d in ds]
    assert len(set(months)) == len(months) and [d.decision_id for d in ds] == [f"d{m}" for m in months]
    for d in ds:
        earlier = [s for s in sess["XNYS"] if rm.eligible[0] <= s < d.session and s.strftime("%Y%m") == d.session.strftime("%Y%m")]
        assert not earlier  # no earlier eligible anchor in its month
        assert d.on_complete_case_row == (d.session in rm.eligible)
        assert d.position == sum(1 for s in rm.eligible if s <= d.session) and d.position >= 1
    assert [d.position for d in ds] == sorted({d.position for d in ds})  # strictly increasing: one model row per decision
    assert ds[0].session >= rm.eligible[0] and ds[-1].session <= rm.eligible[-1]


def test_the_fixture_keeps_every_eligible_row_as_a_decision_and_no_extra_relationship():
    rm, _ = build_row_map(FIX, CALS)
    assert rm.decisions == () and rm.decision_positions == tuple(range(1, 71))
    assert build_split_manifest(FIX, CALS, rm).decisions == ()


def test_owners_cover_every_outer_test_and_the_holdout_without_overlap(maps):
    rm, _, m = maps
    assert [k for k in rm.owners] == [f"outer{i}" for i in range(1, len(m.outer) + 1)] + ["holdout"]
    spans = list(rm.owners.values())
    assert all(a <= b for a, b in spans) and all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))
    assert rm.owners["holdout"][1] == rm.eligible[-1]


def test_the_split_manifest_is_derived_from_the_data_length_with_the_declared_minima(maps):
    rm, _, m = maps
    n = len(rm.eligible)
    assert m.n_positions == n and len(m.outer) == (n - holdout_rows(CFG, n) - 756) // 126 == 5
    lead = outer_leftover_rows(CFG, n)
    assert 0 <= lead < 126 and m.outer[0].train_end == 756 + lead and m.outer[1].test.start_pos == 756 + lead + 126 + 1
    assert all(o.test.n_decisions >= CFG.evaluation.outer.minimum_decisions_per_test for o in m.outer)
    assert m.holdout.n_decisions >= CFG.final_holdout.minimum_decisions and m.holdout.end_pos == m.n_positions
    assert m.refit_positions[0] == 1 and set(m.refit_positions[1:]) == set(range(22, m.n_positions + 1, 21))


def test_the_last_outer_test_ends_on_the_row_before_the_holdout_and_the_leftover_rows_only_extend_the_first_training_span(maps):
    """OQ-048: rows that no whole fold fits go to the first fold's training span, so nothing is carried unscored into the holdout."""
    rm, _, m = maps
    n = len(rm.eligible)
    lead = outer_leftover_rows(CFG, n)
    out = CFG.evaluation.outer
    assert m.outer[-1].test.end_pos + 1 == m.holdout.start_pos
    assert lead == n - holdout_rows(CFG, n) - (out.initial_train_rows + (len(m.outer) - 1) * out.step_rows + out.test_rows)
    assert [b.test.start_pos - a.test.end_pos - 1 for a, b in zip(m.outer, m.outer[1:], strict=False)] == [out.step_rows - out.test_rows] * (len(m.outer) - 1)
    assert m.outer[0].train_end >= out.initial_train_rows  # the declared minimum training history is never reduced


def test_decisions_point_to_one_earlier_or_equal_fit_and_to_one_block(maps):
    rm, _, m = maps
    assert len(m.decisions) == len(rm.decisions)
    for (did, session, position, fit), d in zip(m.decisions, rm.decisions, strict=True):
        assert did == d.decision_id and session == d.session.strftime("%Y-%m-%d") and position == d.position
        assert fit in m.refit_positions and fit <= position < fit + 21  # the last permitted fit, never a later one
    blocks = [o.test for o in m.outer] + [m.holdout]
    owned = [p for b in blocks for p in b.decision_positions]
    assert len(owned) == len(set(owned))  # a decision belongs to at most one scored block


def test_the_decision_set_is_part_of_the_manifest_hash(maps):
    rm, _, m = maps
    changed = type(rm)(rm.anchors, rm.complete_case, rm.warmup, rm.eligible, rm.owners, rm.decisions[1:])
    assert build_split_manifest(CFG, CALS, changed).hash != m.hash


def test_a_snapshot_that_would_need_a_ninth_outer_fold_fails_instead_of_truncating():
    with pytest.raises(ValueError, match=r"requires \d+ outer folds.*maximum_folds is 8"):
        build_row_map(CFG, CALS, ("2008-01-02", "2024-12-31"))


def test_a_snapshot_too_short_for_one_fold_or_the_warmup_is_rejected():
    with pytest.raises(ValueError, match="fewer than 756 training rows"):
        build_row_map(CFG, CALS, ("2016-01-04", "2018-12-31"))
    with pytest.raises(ValueError, match="do not exceed the 252-row warmup"):
        build_row_map(CFG, CALS, ("2016-01-04", "2016-06-30"))


def test_a_snapshot_profile_needs_its_collected_window():
    with pytest.raises(ValueError, match="needs the collected window"):
        build_row_map(CFG, CALS)


def test_the_fold_count_follows_the_ceiling_rule():
    assert outer_fold_count(FIX, 70) == 2  # the fixture runs exactly its declared folds
    assert outer_fold_count(CFG, 1603) == 4  # (1603 - 241 - 756) // 126


def test_a_manifest_cannot_be_built_without_loaded_universe_records():
    bare = load_config(ROOT / "configs" / "research_8_12asset.yaml")
    rm, _ = build_row_map(CFG, CALS, WINDOW)
    with pytest.raises(ValueError, match="universe records are not loaded"):
        build_split_manifest(bare, CALS, rm)


def test_the_interval_report_names_decisions_by_id_and_other_excluded_rows_as_plain_rows(maps):
    """OQ-047: a purged or gap row that is not a decision must not be shown as ``D###``; the censored decision is the block's last decision."""
    from quant_pipeline.intervals import interval_report

    rm, _, m = maps
    report = interval_report(CFG, CALS, rm, m)
    decision_ids = {d.decision_id: d.position for d in rm.decisions}
    seen_rows = seen_decisions = 0
    for b in report["blocks"]:
        for e in b["excluded_observations"]:
            if e["position"] in decision_ids.values():
                assert e["id"] in decision_ids and decision_ids[e["id"]] == e["position"]
                seen_decisions += 1
            else:
                assert e["id"] == f"R{e['position']:04d}" and not e["id"].startswith("D")
                seen_rows += 1
        cid = b["censored_decision"]["id"]
        assert cid in decision_ids and not cid.startswith("D0")
    assert seen_rows > 0 and seen_decisions > 0  # both kinds occur, so neither branch is vacuous
