"""Benchmarks through the path engine (spec 10.3, 10.3.1, 16.10): schedule parity, turnover discipline, staged-entry
attainment and the single labelled exemption. Evidence class: fixture_path."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_phase5_engine as E

from quant_pipeline.benchmarks import make_policy

ENG = E.engine()
END = 40


def bench(bid, start=1, end=END, carry=None, cfg=None):
    pol = make_policy(cfg or E.CFG, bid)
    r = ENG.run(label=bid, fold_id="b", trial=E.TRIAL, trial_hash=f"benchmark:{bid}", multiplier=1.0, start=start, end=end,
                opening=E.OPEN, carry=carry, policy=pol)
    return pol, r


@pytest.fixture(scope="module")
def strat():
    return E.run(ENG, 1, END, label="strategy")


@pytest.fixture(scope="module")
def runs():
    return {bid: bench(bid) for bid in ("cash_krw", "equal_weight_periodic", "equal_weight_staged_entry",
                                         "equal_weight_buy_and_hold_reference")}


def test_every_benchmark_uses_the_strategys_decision_schedule_and_valuation_anchors(runs, strat):
    for bid, (_, r) in runs.items():
        assert [d.cutoff_utc for d in r.decisions] == [d.cutoff_utc for d in strat.decisions], bid  # same decision IDs and cutoffs
        assert list(r.nav) == list(strat.nav), bid  # same canonical valuation timeline


def test_execution_timestamps_match_the_strategy_for_the_same_decision(runs, strat):
    s = {(f.decision_position, f.asset_id): f.ts for f in strat.fills}
    for bid in ("equal_weight_periodic", "equal_weight_staged_entry", "equal_weight_buy_and_hold_reference"):
        for f in runs[bid][1].fills:
            if (f.decision_position, f.asset_id) in s:
                assert f.ts == s[(f.decision_position, f.asset_id)], (bid, f.decision_position, f.asset_id)


def test_cash_reference_never_trades_and_only_accrues(runs):
    _, r = runs["cash_krw"]
    navs = list(r.nav.values())
    assert not r.fills and r.costs == 0 and navs == sorted(navs) and navs[-1] > navs[0] == Decimal("1000000000.00")
    assert all(q == 0 for q in r.carry.runner.holdings.values())


def test_periodic_equal_weight_obeys_the_ordinary_turnover_limit_every_decision(runs):
    _, r = runs["equal_weight_periodic"]
    assert all(d.one_way_turnover_intended <= 0.5 + 1e-9 for d in r.decisions)
    assert r.decisions[0].one_way_turnover_intended == pytest.approx(0.5, abs=2e-3)  # from cash only L can be deployed at once
    assert sum(1 for d in r.decisions if d.n_orders) > 20  # it keeps rebalancing at every shared decision


def test_staged_entry_needs_more_than_one_decision_and_then_holds(runs):
    pol, r = runs["equal_weight_staged_entry"]
    st = pol.state
    assert st["attained"] and st["active_decisions"] >= 2 and not st["stalled"]
    assert r.decisions[0].one_way_turnover_intended == pytest.approx(0.5, abs=2e-3)  # deployment constraint applies
    k = st["attained_at_position"]
    assert all(d.n_orders == 0 for d in r.decisions[k - 1:])  # no voluntary rebalance after the first attainment
    d = r.decisions[k - 1]
    w, tgt = np.array(d.w_minus[:-1]), np.array(d.a_bar[:-1])
    tol = E.CFG.benchmarks.target_attainment_tolerance
    assert np.abs(w - tgt).max() <= tol and abs(w.sum() - tgt.sum()) <= tol and d.pending_at_decision == 0  # 25 bp, nothing pending


def test_staged_entry_reports_a_stall_without_granting_an_exemption():
    cfg = E.CFG.model_copy(update={"benchmarks": E.CFG.benchmarks.model_copy(update={"staged_entry_max_active_decisions": 1})})
    pol, r = bench("equal_weight_staged_entry", end=2, cfg=cfg)
    assert pol.state["stalled"] is True and not pol.state["attained"]
    assert all(d.one_way_turnover_intended <= 0.5 + 1e-9 for d in r.decisions)  # the rule is unchanged: still turnover-limited
    assert r.decisions[1].n_orders > 0  # and it keeps trying


def test_conventional_reference_deploys_once_with_the_only_turnover_exemption(runs):
    pol, r = runs["equal_weight_buy_and_hold_reference"]
    assert pol.state["initial_done"] and pol.state["initial_turnover"] == pytest.approx(0.9)
    assert r.decisions[0].one_way_turnover_intended == pytest.approx(0.9, abs=3e-3)
    assert r.decisions[0].one_way_turnover_intended > E.CFG.constraints.turnover_one_way_limit  # the exemption is real
    assert [d.position for d in r.decisions if d.n_orders] == [1]  # exactly one deployment decision, then no voluntary trades
    p0 = r.plans[0].summary
    assert p0["initial_turnover_exemption"] is True and p0["comparability"] == "reference_initial_turnover_exempt"
    assert all(not r.plans[i].summary.get("initial_turnover_exemption") for i in range(1, len(r.plans)))
    assert Decimal(r.plans[0].projected_post_cost_cash_krw) >= 0  # cash solvency is not waived
    assert r.costs > 0


def test_the_implementation_matched_pair_is_labelled_apart_from_the_reference(runs):
    labels = {bid: r.plans[0].summary.get("comparability") for bid, (_, r) in runs.items() if bid != "cash_krw"}
    assert labels["equal_weight_buy_and_hold_reference"] == "reference_initial_turnover_exempt"
    assert labels["equal_weight_periodic"] is None  # implementation_matched is carried by the benchmark definition and report panel
    defs = {b.benchmark_id: b.comparability for b in E.CFG.benchmarks.definitions if b.type == "rules"}
    assert defs["equal_weight_periodic"] == defs["equal_weight_staged_entry"] == "implementation_matched"
    assert defs["equal_weight_buy_and_hold_reference"] == "reference_initial_turnover_exempt"


def test_all_benchmarks_pay_the_same_cost_model_and_stay_solvent(runs):
    for bid, (_, r) in runs.items():
        assert all(x["within_tolerance"] for x in r.reconciliation), bid
        assert (r.ledger_rows.cash_krw.map(Decimal) >= 0).all(), bid
        assert all(r.ledger_rows[f"qty_{a}"].ge(0).all() for a in ("KRX_A", "KRX_B", "NYSE_C")), bid


def test_staged_entry_state_survives_a_fold_boundary():
    straight = bench("equal_weight_staged_entry", 1, 12)[1]
    pol_a, a = bench("equal_weight_staged_entry", 1, 1)
    pol_b = make_policy(E.CFG, "equal_weight_staged_entry")
    b = ENG.run(label="equal_weight_staged_entry", fold_id="b", trial=E.TRIAL, trial_hash="benchmark:equal_weight_staged_entry",
                multiplier=1.0, start=2, end=12, opening=E.OPEN, carry=a.carry, policy=pol_b)
    assert [d.weights for d in a.decisions + b.decisions] == [d.weights for d in straight.decisions]
    assert pol_b.state == bench("equal_weight_staged_entry", 1, 12)[0].state and pol_a.state["active_decisions"] == 1
