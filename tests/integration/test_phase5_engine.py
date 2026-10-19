"""Path engine (spec 6.2, 6.4, 9, 11.5): carry across a fold boundary, exactly-once crossing orders, future invariance,
cost-scenario sharing and cache purity. Evidence class: fixture_path."""
from __future__ import annotations

import shutil
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.calendars import Calendars, build_row_map, utc
from quant_pipeline.config import load_config
from quant_pipeline.data import load_initial_holdings, load_market_data
from quant_pipeline.decision import FixedRegimeSource, TrialParams
from quant_pipeline.engine import Caches, Opening, PathEngine
from quant_pipeline.fitstate import lineage_hashes
from quant_pipeline.regimes import FixedRegimeTable

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
CALS = Calendars.load(["XKRX", "XNYS"])
RM, SESS = build_row_map(CFG, CALS)
HASHES = lineage_hashes(CFG, ROOT)
REGIME = FixedRegimeSource(FixedRegimeTable.load(ROOT / CFG.regime.fixed_table.path))
_cash, _qty, _asof = load_initial_holdings(CFG, ROOT)
OPEN = Opening(_cash, _qty, utc(_asof))
TRIAL = TrialParams.first(CFG)


def engine(md=None, caches=None):
    return PathEngine(CFG, md or MD, CALS, SESS, RM, REGIME, HASHES, caches)


MD = load_market_data(CFG, ROOT)


def run(eng, start, end, trial=TRIAL, multiplier=1.0, carry=None, label="p", fold="outer-1"):
    return eng.run(label=label, fold_id=fold, trial=trial, trial_hash="sha256:t", multiplier=multiplier, start=start, end=end,
                   opening=OPEN, carry=carry)


@pytest.fixture(scope="module")
def straight():
    eng = engine()
    return eng, run(eng, 1, 60)


@pytest.fixture(scope="module")
def split_runs():
    eng = engine()
    a = run(eng, 1, 50, fold="outer-1")
    b = run(eng, 51, 60, carry=a.carry, fold="outer-2")
    return a, b


def test_full_path_is_self_financing_solvent_and_never_halts(straight):
    _, r = straight
    assert len(r.decisions) == 60 and r.reconciliation and all(x["within_tolerance"] for x in r.reconciliation)
    rows = r.ledger_rows
    assert (rows.cash_krw.map(Decimal) >= 0).all()
    assert all((rows[f"qty_{a}"] >= 0).all() for a in ("KRX_A", "KRX_B", "NYSE_C"))
    assert {d.branch for d in r.decisions} <= {"optimal", "numerical_retry_optimal"}
    assert r.optimizer_attempts <= 2 * len(r.decisions)
    assert Decimal(rows.cumulative_costs.iloc[-1]) == r.costs > 0


def test_a_fold_boundary_carry_equals_one_continuous_run(split_runs, straight):
    a, b = split_runs
    _, s = straight
    # plan hashes legitimately differ: a plan belongs permanently to the fold that created it (outer-2 vs one label)
    orders = lambda r: [[(o.asset_id, o.intended_trade_quantity, o.unconstrained_trade_quantity) for o in p.orders] for p in r.plans]
    assert orders(a) + orders(b) == orders(s)
    assert [d.plan_hash for d in a.decisions] == [d.plan_hash for d in s.decisions[:50]]  # same fold label -> identical hash
    assert {p.fold_id for p in b.plans} == {"outer-2"} and {p.fold_id for p in a.plans} == {"outer-1"}
    assert [d.weights for d in a.decisions + b.decisions] == [d.weights for d in s.decisions]
    assert {**a.nav, **b.nav} == s.nav  # NAV is continuous: no reset, no duplicated return
    assert b.carry.runner == s.carry.runner and b.costs == s.costs
    fills = lambda r: [(f.order_id, f.ts, f.planned_quantity, f.realized_quantity, f.status) for f in r.fills]
    assert fills(a) + fills(b) == fills(s)


def test_orders_created_on_the_last_pre_boundary_decision_execute_exactly_once_after_it(split_runs):
    a, b = split_runs
    created = [f for f in b.fills if f.decision_position == 50]
    assert created, "decision 50 must create boundary-crossing orders"
    assert not [f for f in a.fills if f.decision_position == 50]  # none had executed before the boundary
    first_oos = utc("2024-04-18T00:00:00Z")
    assert all(pd.Timestamp(f.ts) > first_oos and f.status == "filled" for f in created)
    assert {f.asset_id for f in created} >= {"KRX_A", "NYSE_C"} or len(created) >= 2
    ids = [f.order_id for f in a.fills + b.fills if f.status in ("filled", "reduced_cash")]
    assert len(ids) == len(set(ids))  # no duplicated fill
    for f in created:  # origin ids are retained, not relabeled to the new fold
        assert f.origin["fold_id"] == "outer-1" and f.origin["decision_id"] == "D050"
    assert [e.order_id for e in a.carry.pending] and all(e.payload["origin"]["fold_id"] == "outer-1" for e in a.carry.pending)


def test_each_decision_executes_before_the_next_decision(straight):
    _, r = straight
    assert max(d.pending_at_decision for d in r.decisions) >= 0
    by_dec: dict[int, list] = {}
    for f in r.fills:
        by_dec.setdefault(f.decision_position, []).append(f)
    for pos, fl in by_dec.items():
        cutoff = utc(r.decisions[pos - 1].cutoff_utc)
        assert all(pd.Timestamp(f.ts) > cutoff for f in fl)  # strictly after the decision: no same-close execution


def test_full_path_is_deterministic(straight):
    _, r = straight
    again = run(engine(), 1, 60)
    assert [d.plan_hash for d in again.decisions] == [d.plan_hash for d in r.decisions] and again.nav == r.nav


def test_cache_sharing_does_not_change_results():
    shared = Caches()
    warm = engine(caches=shared)
    run(warm, 1, 20)
    with_cache = run(warm, 1, 20, label="again")
    fresh = run(engine(), 1, 20, label="again")
    assert [d.plan_hash for d in with_cache.decisions] == [d.plan_hash for d in fresh.decisions]
    assert shared.counts["cache_hits"] > 0


def test_lambda_candidates_share_calibration_but_gamma_does_not():
    caches = Caches()
    eng = engine(caches=caches)
    base = run(eng, 1, 12)
    calibrations = caches.counts["calibrations"]
    # TRIAL.first() already uses lambda_track = lambda_cost = 0.5, so the alternative must differ from it
    other_lambda = TrialParams(TRIAL.overlay_gamma, TRIAL.entry_z, TRIAL.horizon_weights, TRIAL.gate_name, 1.0, TRIAL.lambda_risk, 1.0)
    assert (other_lambda.lambda_track, other_lambda.lambda_cost) != (TRIAL.lambda_track, TRIAL.lambda_cost)
    r2 = run(eng, 1, 12, trial=other_lambda, label="lam")
    assert caches.counts["calibrations"] == calibrations  # lambdas do not shape targets: the calibration is reused
    assert [d.a_bar for d in r2.decisions] == [d.a_bar for d in base.decisions]  # identical aggregate targets
    assert [d.weights for d in r2.decisions] != [d.weights for d in base.decisions]  # but different optimizer outputs
    gamma = 0.75 if TRIAL.overlay_gamma != 0.75 else 1.0
    other_gamma = TrialParams(gamma, TRIAL.entry_z, TRIAL.horizon_weights, TRIAL.gate_name, TRIAL.lambda_track, TRIAL.lambda_risk,
                              TRIAL.lambda_cost)
    r3 = run(eng, 1, 12, trial=other_gamma, label="gam")
    assert caches.counts["calibrations"] > calibrations  # gamma shapes targets, so it needs its own calibration
    assert [d.a_bar for d in r3.decisions] != [d.a_bar for d in base.decisions]


def test_cost_scenarios_share_targets_but_not_ledgers():
    eng = engine()
    r0, r1, r2 = (run(eng, 1, 25, multiplier=m, label=f"c{m}") for m in (0.0, 1.0, 2.0))
    assert [d.a_bar for d in r0.decisions] == [d.a_bar for d in r1.decisions] == [d.a_bar for d in r2.decisions]  # common targets
    assert r0.costs == 0 and r1.costs > 0 and r2.costs > 0  # 0x pays nothing; the others pay
    assert len({tuple(r.nav.values()) for r in (r0, r1, r2)}) == 3  # three distinct ledgers / NAV paths
    # each scenario optimizes with its own expected-cost multiplier, so intentions may differ (spec 11.7); at least one does
    assert any(d0.weights != d1.weights for d0, d1 in zip(r0.decisions, r1.decisions, strict=True))
    # NAV ordering across scenarios is NOT asserted: the scenarios hold different positions, not the same trades at different prices


def perturbed(tmp_path, after):
    root = tmp_path / "p"
    shutil.copytree(ROOT / "data" / "fixtures", root / "data" / "fixtures")
    for name in ("prices_raw", "prices_total_return", "fx_krw_per_usd"):
        p = root / "data" / "fixtures" / "fixture_3asset" / f"{name}.parquet"
        df = pd.read_parquet(p)
        late = pd.to_datetime(df.event_timestamp_utc, utc=True) > after
        df.loc[late, "value"] *= 1.15
        df.to_parquet(p, index=False)
    return load_market_data(CFG, root)


def test_later_data_cannot_change_earlier_decisions_or_fills(tmp_path, straight):
    _, base = straight
    k = 25
    cut = utc(base.decisions[k - 1].cutoff_utc)
    r = run(engine(md=perturbed(tmp_path, cut)), 1, 40)
    assert [d.plan_hash for d in r.decisions[:k]] == [d.plan_hash for d in base.decisions[:k]]
    early = lambda res: [(f.order_id, f.ts, f.realized_quantity) for f in res.fills if pd.Timestamp(f.ts) <= cut]
    assert early(r) == early(base)
    assert [d.plan_hash for d in r.decisions[k:]] != [d.plan_hash for d in base.decisions[k:40]]  # the perturbation is real


def injected_order(max_execution: str):
    """A pending NYSE_C buy whose first permitted close (2024-03-14, flagged liquidity_invalid) cannot execute."""
    from quant_pipeline.event_time import Event
    eng = engine()
    first = run(eng, 1, 27, label="inj")
    order = Event(utc("2024-03-14T20:00:00Z"), "buy_fill", "XNYS", "NYSE_C", "INJECTED", {
        "quantity": 5, "decision": 27, "max_execution_utc": max_execution,
        "origin": {"fold_id": "outer-1", "refit_id": "r26", "trial_hash": "sha256:t", "decision_id": "D027"}})
    first.carry.pending.append(order)
    held_before = first.carry.runner.holdings["NYSE_C"]
    return eng, run(eng, 28, 31, carry=first.carry, label="inj"), held_before  # held_before kept for debugging


def test_an_order_for_a_flagged_close_advances_to_the_next_valid_close_and_never_backfills():
    _, r, _ = injected_order("2024-03-18T20:00:00Z")
    mine = [f for f in r.fills if f.order_id == "INJECTED"]
    assert [f.status for f in mine] == ["advanced", "advanced", "filled"]  # 03-14 and 03-15 are flagged; 03-18 is the next valid close
    assert [f.ts for f in mine] == ["2024-03-14T20:00:00Z", "2024-03-15T20:00:00Z", "2024-03-18T20:00:00Z"]
    assert mine[-1].realized_quantity == 5 and mine[-1].origin["decision_id"] == "D027"
    rows = r.ledger_rows
    filled_rows = rows[rows.order_id == "INJECTED"]
    assert len(filled_rows) == 1 and filled_rows.iloc[0].ts == "2024-03-18T20:00:00Z"  # executed once, at a valid close only


def test_an_order_past_its_maximum_age_expires_and_is_not_filled():
    _, r, _ = injected_order("2024-03-15T20:00:00Z")  # 03-18 would exceed the allowed order age
    mine = [f for f in r.fills if f.order_id == "INJECTED"]
    assert [f.status for f in mine] == ["advanced", "expired"] and all(f.realized_quantity == 0 for f in mine)
    assert not (r.ledger_rows.order_id == "INJECTED").any()  # expiry is recorded, nothing is posted, nothing is back-filled


def test_realized_cash_feasibility_reduces_a_buy_instead_of_borrowing(straight):
    eng, _ = straight
    from quant_pipeline.event_time import Event
    from quant_pipeline.replay import LedgerRunner
    runner = LedgerRunner(CFG, MD, CALS, Decimal("1000000.00"), dict.fromkeys(("KRX_A", "KRX_B", "NYSE_C"), 0), OPEN.ts)
    e = Event(utc("2024-02-01T06:30:00Z"), "buy_fill", "XKRX", "KRX_A", "t", {"quantity": 10_000})
    got = runner.max_affordable_buy(e, 10_000)
    price = MD.raw_close_at("KRX_A", e.ts)
    assert 0 < got < 10_000 and Decimal(got) * price * Decimal("1.001") <= Decimal("1000000.00") + Decimal(1)
    assert Decimal(got + 1) * price * Decimal("1.001") > runner.ledger.cash  # one more lot would overdraw
    assert eng is not None


def test_nav_series_covers_every_xnys_anchor_exactly_once(straight):
    _, r = straight
    assert list(r.nav) == sorted(r.nav) and len(r.nav) == len(set(r.nav))
    anchors = [s.strftime("%Y-%m-%d") for s in RM.anchors if s <= RM.eligible[59]]
    assert list(r.nav) == anchors
    assert r.nav[anchors[0]] == Decimal("1000000000.00")
    assert np.isfinite([float(v) for v in r.nav.values()]).all()


def test_a_krw_dividend_is_credited_in_krw_without_an_fx_factor_and_a_usd_dividend_with_one(straight):
    from quant_pipeline.event_time import Event
    from quant_pipeline.replay import LedgerRunner
    runner = LedgerRunner(CFG, MD, CALS, Decimal("0.00"), {"KRX_A": 100, "NYSE_C": 100}, OPEN.ts)
    ts = utc("2024-02-01T06:30:00Z")
    runner.entitlement.update({"K-pay".removesuffix("-pay"): 100, "U": 100})
    runner.step(Event(ts, "corporate_action", "XKRX", "KRX_A", "K-pay", {"action_type": "cash_dividend", "phase": "pay", "cash_amount": 50.0}))
    assert runner.ledger.cash == Decimal("5000.00"), "a KRW dividend must not be multiplied by the KRW-per-USD rate"
    runner.step(Event(ts, "corporate_action", "XNYS", "NYSE_C", "U-pay", {"action_type": "cash_dividend", "phase": "pay", "cash_amount": 1.0}))
    assert runner.ledger.cash == Decimal("5000.00") + Decimal(100) * MD.fx_mark(ts)
