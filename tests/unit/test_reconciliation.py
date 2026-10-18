"""Corporate-action reconciliation (spec 4.3, INV-DATA-002): golden arithmetic, limit boundaries, decomposition, halt scope, ordering."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_pipeline import pipeline as pipeline_module
from quant_pipeline import reconciliation as rc
from quant_pipeline.config import load_config
from quant_pipeline.data_quality import validate_data

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
REC = CFG.accounting.corporate_action_reconciliation  # split 1e-8, per event 5 bps, rolling 25 bps
LIMITS = SimpleNamespace(per_event_bps=5.0, rolling_window_bps=25.0, split_relative_tolerance=1e-8)


# ------------------------------------------------------------------------------------------------------ helpers
def frame(raw, tr, sessions=None):
    n = len(raw)
    sessions = sessions or [f"2024-01-{i + 2:02d}" for i in range(n)]
    ts = [pd.Timestamp(f"{s}T06:30:00Z") for s in sessions]
    return pd.DataFrame({"event_timestamp_utc": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in ts], "exchange_session": sessions,
                         "raw": raw, "tr": tr, "ts": ts})


def event(kind, session, *, num=1, den=1, cash=0.0, pay=None, inst="X", action_id="CA-1"):
    return rc._Event(action_id, inst, kind, session, pd.Timestamp(f"{session}T06:30:00Z"),
                     pd.Timestamp(f"{pay}T06:30:00Z") if pay else None, num, den, cash)


def fixture_copy(tmp_path):
    shutil.copytree(ROOT / "data" / "fixtures", tmp_path / "data" / "fixtures")
    return tmp_path


def shift(root, instrument, from_session, bps):
    """Multiply the instrument's total-return index by (1 + bps) from ``from_session`` on: that session's return is ``bps`` off, later ones intact."""
    path = root / CFG.data.prices_total_return
    df = pd.read_parquet(path)
    df.loc[(df.instrument_id == instrument) & (df.exchange_session >= from_session), "value"] *= 1 + bps / 1e4
    df.to_parquet(path)


# ------------------------------------------------------------------------------------------------------ level 1
def test_a_whole_share_split_has_no_residual_and_scales_exactly():
    assert rc.split_quantity(1000, 2, 1, 1e-8) == 2000
    assert rc.split_quantity(1002, 3, 2, 1e-8) == 1503
    assert rc.split_quantity(0, 3, 2, 1e-8) == 0


def test_a_fractional_share_above_tolerance_halts_and_below_tolerance_rounds_toward_zero():
    with pytest.raises(rc.ReconciliationHalt, match="level 1"):
        rc.split_quantity(1001, 3, 2, 1e-8)  # 1501.5 shares: relative residual 3.3e-4
    big = 10**12 + 1  # 1.5e12 + 1.5 shares: residual 1e-12 relative, inside the declared tolerance
    assert rc.split_quantity(big, 3, 2, 1e-8) == (big * 3) // 2


def test_an_invalid_ratio_or_a_negative_quantity_cannot_be_split():
    for args in ((10, 0, 1), (10, 2, 0), (10, -2, 1), (-1, 2, 1)):
        with pytest.raises(rc.ReconciliationHalt):
            rc.split_quantity(*args, 1e-8)


# ------------------------------------------------------------------------------------------------------ level 2
def test_golden_reconstruction_by_hand_for_a_split_followed_by_a_dividend():
    # raw 100, then 52 after a 2:1 split (adjusted 104 = +4%), then 54 (adjusted 108) with a 0.52 per-share dividend (adjusted 1.04)
    ev = [event("split", "2024-01-03", num=2, den=1, action_id="CA-S"), event("cash_dividend", "2024-01-04", cash=0.52, action_id="CA-D")]
    f = frame([100.0, 52.0, 54.0], [100.0, 104.0, 109.04])
    got = rc.reconstruct_total_return(f, ev, "X")
    assert got == pytest.approx([100.0, 104.0, 100.0 * (104.0 / 100.0) * (108.0 + 1.04) / 104.0], rel=1e-12)
    assert got[2] == pytest.approx(109.04, rel=1e-12)
    breaches, diag, _ = rc._level2(f, ev, "X", LIMITS)
    assert breaches == [] and diag["max_per_event_bps"] < 1e-9


def test_other_instruments_actions_never_affect_an_instrument():
    f = frame([100.0, 101.0, 102.0], [100.0, 101.0, 102.0])
    ev = [event("split", "2024-01-03", num=2, den=1, inst="OTHER"), event("cash_dividend", "2024-01-04", cash=5.0, inst="OTHER")]
    assert rc.reconstruct_total_return(f, ev, "X") == pytest.approx([100.0, 101.0, 102.0])


def test_the_real_fixture_reconciles_at_every_level():
    rep = rc.reconcile(CFG, ROOT)
    assert rep.breaches == [] and not rep.halt and rep.scope == "none"
    assert all(r["status"] == "pass" for r in rep.results())
    assert max(d["max_per_event_bps"] for d in rep.diagnostics.values()) < 0.01  # only index rounding remains
    assert max(d["level3_max_unexplained_bps"] for d in rep.diagnostics.values()) < 0.01


def test_a_per_event_residual_just_below_the_limit_passes_and_just_above_fails(tmp_path):
    ok = fixture_copy(tmp_path / "ok")
    shift(ok, "KRX_B", "2024-03-05", 4.0)  # the split's ex session is 4 bps off against a 5 bps limit
    assert rc.reconcile(CFG, ok).breaches == []
    bad = fixture_copy(tmp_path / "bad")
    shift(bad, "KRX_B", "2024-03-05", 10.0)  # 10 bps off
    rep = rc.reconcile(CFG, bad)
    assert {(b.level, b.instrument_id, b.kind) for b in rep.breaches} >= {(2, "KRX_B", "per_event")}
    assert rep.invalidated["KRX_B"].startswith("2024-03-05")  # invalid from its first offending event
    assert rep.halt and rep.scope == "run" and "fixture" in rep.reason
    with pytest.raises(rc.ReconciliationHalt, match="halts the run"):
        rc.require_clean(CFG, bad)


def test_a_slow_drift_trips_the_rolling_window_without_any_per_event_breach(tmp_path):
    fixture_copy(tmp_path)
    sessions = sorted(pd.read_parquet(tmp_path / CFG.data.prices_total_return).query("instrument_id == 'KRX_A'").exchange_session)[10:20]
    for sess in sessions:  # ten non-event sessions, each 3 bps off: 30 bps cumulative against a 25 bps limit
        shift(tmp_path, "KRX_A", sess, 3.0)
    rep = rc.reconcile(CFG, tmp_path)
    assert {b.kind for b in rep.breaches if b.level == 2} == {"rolling_window"}
    assert any(b.level == 3 for b in rep.breaches)  # the decomposition cannot explain a persistent gap either
    assert rep.halt


def test_an_unsupported_action_halts_before_the_event(tmp_path):
    fixture_copy(tmp_path)
    path = tmp_path / CFG.data.corporate_actions
    ca = pd.read_parquet(path)
    row = ca.iloc[0].copy()
    row["action_id"], row["action_type"] = "CA-0009", "merger"
    pd.concat([ca, row.to_frame().T], ignore_index=True).to_parquet(path)
    rep = rc.reconcile(CFG, tmp_path)
    assert any(b.kind == "unsupported_action" for b in rep.breaches) and rep.halt


# ------------------------------------------------------------------------------------------------------ level 3
def test_dividend_timing_and_non_reinvestment_are_explained_not_flagged():
    diag = rc.reconcile(CFG, ROOT).diagnostics["NYSE_C"]
    assert diag["level3_max_timing_bps"] > 30.0 and diag["level3_max_non_reinvestment_bps"] > 1.0  # far above both limits ...
    assert diag["level3_max_unexplained_bps"] < 0.01  # ... but explained, so no breach


def test_a_gap_the_decomposition_cannot_explain_is_reported_at_level_3(tmp_path):
    fixture_copy(tmp_path)
    shift(tmp_path, "NYSE_C", "2024-02-20", 60.0)  # a persistent +60 bps level shift on a non-event day
    rep = rc.reconcile(CFG, tmp_path)
    b3 = [b for b in rep.breaches if b.level == 3]
    assert b3 and b3[0].instrument_id == "NYSE_C" and abs(b3[0].residual_bps) > 25.0


# ------------------------------------------------------------------------------------------------------ halt scope
def scope(*, fixture=False, universe=("A", "B", "C", "D"), universe_min=3, held=(), pending=(), bench=(), rule=None, level=2, assets=("D",),
          types=None):
    rep = rc.ReconciliationReport()
    rep.breaches = [rc.Breach(level, a, "per_event", "2024-03-05T06:30:00+00:00", 9.0, 5.0, "x") for a in assets]
    rep.invalidated = {a: "2024-03-05T06:30:00+00:00" for a in assets}
    return rc.halt_scope(rep, fixture=fixture, universe=set(universe), universe_min=universe_min, held=set(held), pending=set(pending),
                         benchmark_required=set(bench), event_types=types or {a: "cash_dividend" for a in assets}, rule=rule)


RULE = rc.ExclusionRule.parse({"event_types": ["cash_dividend"], "effective_cutoff_utc": "2024-01-01T00:00:00Z", "min_universe": 3,
                               "benchmark_consequence": "asset removed from rule benchmarks", "frozen_before_evaluation": True})


def test_the_halt_scope_decisions():
    assert scope(assets=())[0] is False  # no breach, no halt
    assert scope(fixture=True, rule=RULE) == (True, "run", "the fixture has no exclusion rule: every reconciliation breach halts the entire run")
    assert scope(rule=None)[0] is True  # research without a frozen rule: halt
    assert scope(rule=RULE)[:2] == (False, "asset")  # frozen rule, asset free, universe stays at 3
    assert scope(rule=RULE, held={"D"})[0] is True
    assert scope(rule=RULE, pending={"D"})[0] is True
    assert scope(rule=RULE, bench={"D"})[0] is True
    assert scope(rule=RULE, universe=("A", "B", "D"))[0] is True  # removal would leave 2 < 3
    assert scope(rule=RULE, types={"D": "split"})[0] is True  # event type not covered by the rule
    late = rc.ExclusionRule.parse({"event_types": ["cash_dividend"], "min_universe": 3, "benchmark_consequence": "c",
                                      "frozen_before_evaluation": True, "effective_cutoff_utc": "2024-06-01T00:00:00Z"})
    assert scope(rule=late)[0] is True  # the offending event precedes the rule's effective cutoff
    assert scope(rule=RULE, level=1)[0] is True  # a level-1 breach always halts the entire run


def test_an_incomplete_or_unfrozen_exclusion_rule_cannot_exclude_anything():
    base = {"event_types": ["cash_dividend"], "effective_cutoff_utc": "2024-01-01T00:00:00Z", "min_universe": 3,
            "benchmark_consequence": "c", "frozen_before_evaluation": True}
    assert rc.ExclusionRule.parse(None) is None
    for missing in base:
        with pytest.raises(rc.ReconciliationHalt, match="missing"):
            rc.ExclusionRule.parse({k: v for k, v in base.items() if k != missing})
    with pytest.raises(rc.ReconciliationHalt, match="retrospective"):
        rc.ExclusionRule.parse({**base, "frozen_before_evaluation": False})


# ------------------------------------------------------------------------------------------------------ integration
def test_data_validate_reports_all_reconciliation_checks():
    rows = {r["check"]: r["status"] for r in validate_data(CFG, ROOT)}
    for name in ("ca_reconciliation_level1_split_ledger", "ca_reconciliation_level2_total_return",
                 "ca_reconciliation_level3_executable_vs_adjusted", "ca_reconciliation_halt_scope"):
        assert rows[name] == "pass"


def test_a_breach_halts_the_pipeline_before_any_market_data_or_decision(monkeypatch, tmp_path):
    called = []

    def boom(cfg, root, **kw):
        raise rc.ReconciliationHalt("breach")
    monkeypatch.setattr(pipeline_module, "require_clean", boom)
    monkeypatch.setattr(pipeline_module, "load_market_data", lambda *a, **k: called.append("market data"))
    with pytest.raises(rc.ReconciliationHalt):
        pipeline_module.FixturePipeline(ROOT, tmp_path / "out", "r1")
    assert called == [] and not (tmp_path / "out").exists()  # nothing was loaded, decided or checkpointed


def test_a_corrupt_series_is_reported_as_a_breach_not_a_crash(tmp_path):
    fixture_copy(tmp_path)
    path = tmp_path / CFG.data.prices_raw
    df = pd.read_parquet(path)
    df.loc[3, "value"] = -1.0
    df.to_parquet(path)
    rep = rc.reconcile(CFG, tmp_path)
    assert rep.halt and any(b.kind == "series" and "non-positive" in b.detail for b in rep.breaches)
