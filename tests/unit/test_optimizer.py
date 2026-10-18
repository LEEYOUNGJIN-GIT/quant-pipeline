"""Optimizer: feasibility, independent optimality, constraint binding, OSQP status handling and failure policy
(spec 8.4, 8.5, 9.1, 9.5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from quant_pipeline.config import load_config
from quant_pipeline.objective import objective_terms
from quant_pipeline.optimizer import (
    OptimizerInputs,
    SolveOutcome,
    SpecificationError,
    hard_constraint_violations,
    optimize,
    proportional_sells,
    reachable_min_gross,
    solve_qp,
)

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
OK = {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 20000}


def inputs(**over) -> OptimizerInputs:
    base = {
        "a_bar": np.array([0.30, 0.20, 0.10, 0.40]), "w_minus": np.array([0.0, 0.0, 0.0, 1.0]),
        "sigma": np.diag([0.04, 0.01, 0.02]), "w_track": np.ones(4), "scales": (0.02, 0.005, 0.001), "lambdas": (1.0, 1.0, 1.0),
        "rates": np.array([0.001, 0.001, 0.0012, 0.0]), "b_final": 0.6, "target_cash_floor": 0.05,
        "max_weights": np.full(3, 0.6), "usd_mask": np.array([False, False, True]), "usd_limit": 0.5, "turnover_limit": 0.5,
        "tradable": np.array([True, True, True]),
    }
    return OptimizerInputs(**(base | over))


def independent_optimum(inp: OptimizerInputs):
    """Same problem as an NLP with explicit L1 auxiliary variables, solved by SLSQP: a different formulation and solver."""
    n = inp.n
    lt, lr, lc = inp.lambdas
    st, sr, sc = inp.scales

    def f(x):
        w, t = x[: n + 1], x[n + 1:]
        d = w - inp.a_bar
        return lt / st * d @ (inp.w_track * d) + lr / sr * w[:n] @ inp.sigma @ w[:n] + lc / sc * inp.rates @ t

    cons = [{"type": "eq", "fun": lambda x: x[: n + 1].sum() - 1},
            {"type": "ineq", "fun": lambda x: x[n] - inp.target_cash_floor},
            {"type": "ineq", "fun": lambda x: inp.b_final - x[:n].sum()},
            {"type": "ineq", "fun": lambda x: inp.turnover_limit - 0.5 * x[n + 1:].sum()},
            {"type": "ineq", "fun": lambda x: x[n + 1:] - (x[: n + 1] - inp.w_minus)},
            {"type": "ineq", "fun": lambda x: x[n + 1:] - (inp.w_minus - x[: n + 1])},
            {"type": "ineq", "fun": lambda x: inp.max_weights - x[:n]},
            {"type": "ineq", "fun": lambda x: inp.usd_limit - x[:n][inp.usd_mask].sum()}]
    x0 = np.r_[inp.w_minus, np.zeros(n + 1)]
    r = minimize(f, x0, method="SLSQP", constraints=cons, bounds=[(0, 1)] * (n + 1) + [(0, 2)] * (n + 1),
                 options={"ftol": 1e-14, "maxiter": 500})
    return r.x[: n + 1], r.fun


def test_solution_is_feasible_and_matches_an_independent_slsqp_optimum():
    inp = inputs()
    r = optimize(inp, CFG)
    assert r.branch == "optimal" and r.status == "optimal" and len(r.attempts) == 1
    assert max(hard_constraint_violations(r.weights, inp).values()) <= 1e-8
    w_ref, f_ref = independent_optimum(inp)
    assert r.terms.objective == pytest.approx(f_ref, rel=1e-5)
    np.testing.assert_allclose(r.weights, w_ref, atol=2e-3)


def test_objective_contributions_reconcile_with_an_independent_recompute():
    inp = inputs()
    r = optimize(inp, CFG)
    t = objective_terms(r.weights, inp.a_bar, inp.w_minus, inp.w_track, inp.sigma, inp.rates, inp.scales, inp.lambdas)
    assert sum(r.terms.contributions) == pytest.approx(t.objective) == pytest.approx(r.terms.objective)
    assert r.terms.contributions == pytest.approx(t.contributions) and all(c >= 0 for c in r.terms.contributions)


@pytest.mark.parametrize("name,over,check", [
    ("cash_floor", {"a_bar": np.array([0.4, 0.4, 0.2, 0.0]), "b_final": 0.95, "max_weights": np.full(3, 0.9),
                    "usd_limit": 1.0, "turnover_limit": 1.0, "lambdas": (50.0, 0.001, 0.001)},
     lambda w: w[3] == pytest.approx(0.05, abs=1e-6)),
    ("b_final", {"a_bar": np.array([0.4, 0.4, 0.2, 0.0]), "b_final": 0.30, "turnover_limit": 1.0, "lambdas": (50.0, 0.001, 0.001)},
     lambda w: w[:3].sum() == pytest.approx(0.30, abs=1e-6)),
    ("asset_cap", {"a_bar": np.array([0.9, 0.0, 0.0, 0.1]), "b_final": 0.9, "turnover_limit": 1.0, "lambdas": (50.0, 0.001, 0.001)},
     lambda w: w[0] == pytest.approx(0.6, abs=1e-6)),
    ("usd_limit", {"a_bar": np.array([0.0, 0.0, 0.6, 0.4]), "usd_limit": 0.2, "turnover_limit": 1.0, "lambdas": (50.0, 0.001, 0.001)},
     lambda w: w[2] == pytest.approx(0.2, abs=1e-6)),
    ("turnover", {"a_bar": np.array([0.3, 0.3, 0.0, 0.4]), "turnover_limit": 0.2, "lambdas": (50.0, 0.001, 0.001)},
     lambda w: 0.5 * np.abs(w - np.array([0, 0, 0, 1.0])).sum() == pytest.approx(0.2, abs=1e-6)),
])
def test_each_hard_constraint_binds_when_the_target_pushes_against_it(name, over, check):
    inp = inputs(**over)
    r = optimize(inp, CFG)
    # stiff cases may need the one tighter retry because the independent recompute (1e-8) is stricter than eps 1e-7
    assert r.branch in ("optimal", "numerical_retry_optimal") and len(r.attempts) <= 2 and check(r.weights), (name, r.weights)
    assert max(hard_constraint_violations(r.weights, inp).values()) <= 1e-8


def test_an_untradable_asset_keeps_its_marked_weight_exactly():
    inp = inputs(w_minus=np.array([0.0, 0.2, 0.0, 0.8]), tradable=np.array([True, False, True]),
                 a_bar=np.array([0.2, 0.5, 0.1, 0.2]))
    r = optimize(inp, CFG)
    assert r.weights[1] == pytest.approx(0.2, abs=1e-9) and r.branch in ("optimal", "numerical_retry_optimal")


def test_reachable_gross_precheck_arithmetic():
    inp = inputs(w_minus=np.array([0.3, 0.2, 0.2, 0.3]), tradable=np.array([True, False, True]), turnover_limit=0.1)
    lo, g, locked = reachable_min_gross(inp)
    assert (g, locked) == (pytest.approx(0.7), pytest.approx(0.2)) and lo == pytest.approx(0.6)  # max(locked, g - L)


# ---------------------------------------------------------------------------------- status handling (spec 8.5)
def outcome(status, w=None, prim=1e-9, dual=1e-9, pcert=False, dcert=False):
    return SolveOutcome(status, w, prim, dual, pcert, dcert, OK)


def real_then(inp, first):
    calls = []

    def solve(i, s):
        calls.append(s)
        return first if len(calls) == 1 else solve_qp(i, s)
    return solve, calls


def test_optimal_inaccurate_is_accepted_only_with_small_residuals():
    inp = inputs()
    good = optimize(inp, CFG).weights
    ok = optimize(inp, CFG, solve=lambda i, s: outcome("optimal_inaccurate", good, 5e-9, 5e-8))
    assert ok.branch == "optimal_inaccurate_accepted" and len(ok.attempts) == 1
    bad_primal = optimize(inp, CFG, solve=lambda i, s: outcome("optimal_inaccurate", good, 5e-8, 5e-8))
    assert bad_primal.branch != "optimal_inaccurate_accepted" and len(bad_primal.attempts) == 2
    bad_dual = optimize(inp, CFG, solve=lambda i, s: outcome("optimal_inaccurate", good, 5e-9, 5e-6))
    assert bad_dual.branch != "optimal_inaccurate_accepted"


def test_a_numerical_failure_triggers_exactly_one_tighter_retry():
    inp = inputs()
    solve, calls = real_then(inp, outcome("solver_error"))
    r = optimize(inp, CFG, solve=solve)
    assert r.branch == "numerical_retry_optimal" and len(calls) == 2 and len(r.attempts) == 2
    assert calls[1]["eps_abs"] < calls[0]["eps_abs"] and calls[1]["max_iter"] > calls[0]["max_iter"] and calls[1]["polishing"]


def test_attempts_per_problem_are_bounded_by_the_profile_limit():
    n = []
    r = optimize(inputs(), CFG, solve=lambda i, s: n.append(1) or outcome("solver_error"), previous_target=None, first_decision=True)
    assert len(n) == CFG.compute_budget.max_solver_attempts_per_problem == 2
    assert r.branch == "use_marked_holdings"


def test_constraint_violating_solution_is_rejected_even_if_the_solver_says_optimal():
    inp = inputs()
    cheat = np.array([0.5, 0.5, 0.5, -0.5])  # sums to one but violates long-only, caps and the cash floor
    r = optimize(inp, CFG, solve=lambda i, s: outcome("optimal", cheat), first_decision=True)
    assert r.branch == "use_marked_holdings" and not any(a["accepted"] for a in r.attempts)


def test_fallback_chain_hold_previous_marked_holdings_emergency_and_halt():
    fail = lambda i, s: outcome("solver_error")
    prev = np.array([0.2, 0.2, 0.1, 0.5])
    held = optimize(inputs(w_minus=np.array([0.1, 0.1, 0.1, 0.7])), CFG, solve=fail, previous_target=prev)
    assert held.branch == "hold_previous" and np.array_equal(held.weights, prev)
    over = optimize(inputs(w_minus=np.array([0.4, 0.2, 0.1, 0.3]), b_final=0.55, turnover_limit=1.0), CFG, solve=fail)
    assert over.branch == "emergency_sell_only" and over.weights[:3].sum() == pytest.approx(0.55)
    assert (over.weights[:3] <= np.array([0.4, 0.2, 0.1]) + 1e-12).all()  # sell-only: no buys
    # previous target exceeds B_final but the marked holdings are fine: do not trade (not a mislabelled "emergency")
    keep = optimize(inputs(w_minus=np.array([0.1, 0.1, 0.1, 0.7])), CFG, solve=fail, previous_target=np.array([0.5, 0.3, 0.1, 0.1]))
    assert keep.branch == "use_marked_holdings" and keep.turnover == 0.0
    # marked holdings themselves breach a per-asset cap (drift) and the solver failed: no self-financing path -> halt
    still_valid = optimize(inputs(w_minus=np.array([0.7, 0.0, 0.0, 0.3]), b_final=0.9), CFG, solve=fail,
                           previous_target=np.array([0.5, 0.3, 0.1, 0.1]))
    assert still_valid.branch == "hold_previous"  # a valid previous target is held even though the marked state drifted
    halted = optimize(inputs(w_minus=np.array([0.7, 0.0, 0.0, 0.3]), b_final=0.9), CFG, solve=fail, previous_target=None)
    assert halted.halted and halted.weights is None and halted.halt_reason == "no_self_financing_path"
    assert "marked holdings violate asset_caps" in halted.notes


def test_confirmed_primal_infeasibility_takes_the_cash_only_branch():
    # needs cash >= 0.65 from a 0.7-risky portfolio but only 0.30 of turnover is allowed -> infeasible
    inp = inputs(w_minus=np.array([0.3, 0.2, 0.2, 0.3]), target_cash_floor=0.65, turnover_limit=0.30, b_final=0.9)
    r = optimize(inp, CFG)
    assert r.branch == "cash_only" and r.emergency["reason"] == "confirmed_primal_infeasible"
    assert r.attempts[0]["status"] in ("infeasible", "infeasible_inaccurate") and r.attempts[0]["infeasibility_confirmed"]
    assert r.weights[:3].sum() == 0.0 and r.weights[3] == pytest.approx(1.0) and not r.halted
    assert r.turnover > inp.turnover_limit  # ordinary turnover does not block forced sells


def test_infeasible_without_a_certificate_is_a_numerical_failure_not_a_liquidation():
    inp = inputs()
    r = optimize(inp, CFG, solve=lambda i, s: outcome("infeasible", None, pcert=False), first_decision=True)
    assert r.branch == "use_marked_holdings" and all(not a.get("infeasibility_confirmed", False) for a in r.attempts)


def test_dual_infeasibility_halts_with_a_specification_error():
    with pytest.raises(SpecificationError):
        optimize(inputs(), CFG, solve=lambda i, s: outcome("unbounded", None, dcert=True))
    with pytest.raises(SpecificationError):
        optimize(inputs(), CFG, solve=lambda i, s: outcome("unbounded_inaccurate", None))


# ---------------------------------------------------------------------------------- emergency de-risking and cash_only
def test_emergency_exceeds_ordinary_turnover_only_by_the_minimum_required():
    inp = inputs(w_minus=np.array([0.3, 0.3, 0.2, 0.2]), b_final=0.4, turnover_limit=0.2)
    r = optimize(inp, CFG)
    assert r.branch == "emergency_sell_only" and not r.halted
    assert r.emergency["required_turnover"] == pytest.approx(0.4) and r.emergency["applied_turnover"] == pytest.approx(0.4)
    assert r.turnover > inp.turnover_limit and r.weights[:3].sum() == pytest.approx(0.4)
    assert (r.weights[:3] <= inp.w_minus[:3] + 1e-12).all() and r.weights.sum() == pytest.approx(1.0)


def test_emergency_with_an_untradable_residual_halts_after_preserving_state():
    inp = inputs(w_minus=np.array([0.3, 0.3, 0.2, 0.2]), tradable=np.array([True, False, True]), b_final=0.2, turnover_limit=0.1)
    r = optimize(inp, CFG)
    assert r.branch == "emergency_sell_only" and r.halted and r.halt_reason == "untradable_residual"
    assert r.weights[1] == pytest.approx(0.3) and r.emergency["untradable_residual"] == pytest.approx(0.1)


def test_proportional_sells_never_buy_and_keep_locked_positions():
    inp = inputs(w_minus=np.array([0.4, 0.2, 0.2, 0.2]), tradable=np.array([True, True, False]))
    w = proportional_sells(inp, 0.5)
    assert w[2] == 0.2 and w[:2].sum() == pytest.approx(0.3) and (w[:3] <= inp.w_minus[:3] + 1e-12).all()
    np.testing.assert_allclose(w[0] / w[1], 2.0)  # pro rata
    assert w.sum() == pytest.approx(1.0)
