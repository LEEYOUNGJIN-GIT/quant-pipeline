"""Split manifest (spec 11.1, 11.2, 14): geometry fixed by the spec text, fences, ownership and minimum-count assertions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_pipeline.calendars import Calendars, build_row_map
from quant_pipeline.config import load_config
from quant_pipeline.splits import build_split_manifest, purge_range_clash, refit_schedule, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
CALS = Calendars.load(["XKRX", "XNYS"])
RM, _ = build_row_map(CFG, CALS)
M = build_split_manifest(CFG, CALS, RM)


def test_selection_pass_geometry_matches_spec_11_1():
    got = {p.name: [(f.validation.start_pos, f.validation.end_pos) for f in p.inner] for p in M.passes}
    assert got == {"outer-1": [(30, 34), (36, 40)], "outer-2": [(40, 44), (46, 50)], "final": [(50, 54), (56, 60)]}
    assert [p.train_end for p in M.passes] == [40, 50, 60]
    assert M.path_endpoints() == [34, 40, 44, 50, 54, 60]
    for p in M.passes:  # position T-5 is the explicit one-row gap between the two blocks
        first, second = p.inner
        assert first.gap_positions == (p.train_end - 5,) and second.validation.start_pos - first.validation.end_pos == 2


def test_outer_tests_and_holdout_follow_spec_14():
    assert [(o.test.start_pos, o.test.end_pos) for o in M.outer] == [(41, 50), (51, 60)]
    assert (M.holdout.start_pos, M.holdout.end_pos) == (61, 70) and M.n_positions == 70
    assert [o.test.n_valuation_intervals for o in M.outer] == [11, 11] and M.outer_valuation_intervals() == 22
    assert M.holdout.n_valuation_intervals == 12
    assert [RM.eligible[p - 1].strftime("%Y-%m-%d") for p in (41, 50, 51, 60, 61, 70)] == [
        "2024-04-03", "2024-04-17", "2024-04-18", "2024-05-02", "2024-05-03", "2024-05-20"]


def test_valuation_ownership_equals_the_independent_oracle():
    cal = pd.read_csv(ROOT / "tests/expected/fixture_3asset/oracle/calendar_sessions.csv", keep_default_na=False)
    owned = cal[(cal.calendar == "XNYS") & (cal.valuation_owner != "")]
    want = {k: tuple(g.session_date) for k, g in owned.groupby("valuation_owner")}
    assert want["outer1"] == M.outer[0].test.valuation_sessions
    assert want["outer2"] == M.outer[1].test.valuation_sessions and want["holdout"] == M.holdout.valuation_sessions


def test_no_valuation_interval_is_owned_twice_and_holdout_never_enters_oos():
    outer = [s for o in M.outer for s in o.test.valuation_sessions]
    assert len(outer) == len(set(outer)) == 22 and not set(outer) & set(M.holdout.valuation_sessions)
    assert max(outer) < min(M.holdout.valuation_sessions)


def test_every_block_meets_its_minimum_counts_after_event_time_fencing():
    for p in M.passes:
        for f in p.inner:
            assert f.executable_decisions >= 4 and f.validation.n_valuation_intervals >= 4 and f.censored_decision == f.validation.end_pos
    for o in M.outer:
        assert o.test.n_decisions == 10 and o.test.n_valuation_intervals >= 8
    assert M.holdout.n_decisions == 10


def test_the_last_decision_of_a_validation_block_is_censored_by_the_information_fence():
    for p in M.passes:
        for f in p.inner:
            b = f.validation
            assert pd.Timestamp(b.protected_end_utc) > pd.Timestamp(b.fence_utc) > pd.Timestamp(b.first_decision_cutoff_utc)
            assert f.executable_decisions == b.n_decisions - 1


def test_purge_rows_precede_each_block_and_do_not_clash_with_the_refit_schedule():
    for p in M.passes:
        for f in p.inner:
            assert f.purge_positions == tuple(range(f.validation.start_pos - 2, f.validation.start_pos))
            assert purge_range_clash(p.refit_positions, f) == ()
    # a refit inside a purge range would be reported (and must be dropped for that fold)
    f = M.passes[0].inner[0]
    assert purge_range_clash((1, 6, 28), f) == (28,)


def test_refit_positions_follow_the_continuous_schedule():
    assert M.refit_positions[:4] == (1, 6, 11, 16) and M.refit_positions[-1] == 66
    assert refit_schedule(CFG, 12) == (1, 6, 11)
    assert all(r <= p.train_end for p in M.passes for r in p.refit_positions)


def test_manifest_hash_is_stable_and_depends_on_geometry():
    assert build_split_manifest(CFG, CALS, RM).hash == M.hash
    cfg = CFG.model_copy(update={"evaluation": CFG.evaluation.model_copy(update={"purge": CFG.evaluation.purge.model_copy(update={"minimum_rows": 3})})})
    assert build_split_manifest(cfg, CALS, RM).hash != M.hash  # same counts, different purge geometry


def test_invalid_geometry_is_rejected_not_adapted():
    out = CFG.evaluation.outer
    big = CFG.model_copy(update={"evaluation": CFG.evaluation.model_copy(update={"outer": out.model_copy(update={"initial_train_rows": 55})})})
    with pytest.raises(ValueError, match="holdout"):
        build_split_manifest(big, CALS, RM)  # the second test block would overlap the holdout
    thin = CFG.model_copy(update={"evaluation": CFG.evaluation.model_copy(update={"inner": CFG.evaluation.inner.model_copy(update={"validation_rows": 3})})})
    with pytest.raises(ValueError, match="executable decisions"):
        build_split_manifest(thin, CALS, RM)  # 3 rows leave only 2 executable decisions after the censored last one
    wrong = CFG.model_copy(update={"valuation": CFG.valuation.model_copy(update={"expected_outer_oos_intervals": 20})})
    with pytest.raises(ValueError, match="differ from the profile"):
        validate_manifest(wrong, M)
