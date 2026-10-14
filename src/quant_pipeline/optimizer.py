"""Canonical tracking-risk-cost optimizer with explicit OSQP policy, status handling and fallbacks (spec 8.4, 8.5, 9.1, 9.5).

Weights are continuous *intentions*. Holdings arise only through raw-price sizing, lot rounding, next-close fills and the
ledger. Every branch has a distinct status; nothing is silently substituted.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .config import Config
from .objective import Terms, objective_terms, one_way_turnover


class SpecificationError(RuntimeError):
    """Dual infeasibility: an objective/specification error that halts and never invokes liquidation (spec 8.5 item 6)."""


@dataclass(frozen=True)
class OptimizerInputs:
    a_bar: NDArray[np.float64]  # risky then cash
    w_minus: NDArray[np.float64]  # marked pre-trade full weights
    sigma: NDArray[np.float64]
    w_track: NDArray[np.float64]  # diagonal
    scales: tuple[float, float, float]
    lambdas: tuple[float, float, float]
    rates: NDArray[np.float64]  # scenario multiplier already applied
    b_final: float
    target_cash_floor: float
    max_weights: NDArray[np.float64]
    usd_mask: NDArray[np.bool_]
    usd_limit: float
    turnover_limit: float
    tradable: NDArray[np.bool_]

    @property
    def n(self) -> int:
        return len(self.sigma)


@dataclass(frozen=True)
class SolveOutcome:
    status: str
    w: NDArray[np.float64] | None
    prim_res: float | None
    dual_res: float | None
    prim_inf_cert: bool
    dual_inf_cert: bool
    settings: dict[str, Any]


SolveFn = Callable[[OptimizerInputs, dict[str, Any]], SolveOutcome]


def hard_constraint_violations(w: NDArray[np.float64], inp: OptimizerInputs, turnover_limit: float | None = None) -> dict[str, float]:
    """Independent recomputation of every hard constraint; values are the positive violation magnitudes."""
    n, lim = inp.n, inp.turnover_limit if turnover_limit is None else turnover_limit
    v = {
        "budget": abs(float(w.sum() - 1.0)),
        "long_only": float(max(0.0, -w.min())),
        "cash_floor": float(max(0.0, inp.target_cash_floor - w[n])),
        "risky_gross": float(max(0.0, w[:n].sum() - inp.b_final)),
        "asset_caps": float(max(0.0, (w[:n] - inp.max_weights).max())),
        "usd_exposure": float(max(0.0, w[:n][inp.usd_mask].sum() - inp.usd_limit)),
        "turnover": float(max(0.0, one_way_turnover(w, inp.w_minus) - lim)),
        "untradable_held": float(np.abs(w[:n][~inp.tradable] - inp.w_minus[:n][~inp.tradable]).max(initial=0.0)),
    }
    return v


def solve_qp(inp: OptimizerInputs, settings: dict[str, Any]) -> SolveOutcome:
    n = inp.n
    w = cp.Variable(n + 1)
    lt, lr, lc = inp.lambdas
    st, sr, sc = inp.scales
    obj = (lt / st * cp.quad_form(w - inp.a_bar, cp.psd_wrap(np.diag(inp.w_track)))
           + lr / sr * cp.quad_form(w[:n], cp.psd_wrap(inp.sigma))
           + lc / sc * (inp.rates @ cp.abs(w - inp.w_minus)))
    cons = [cp.sum(w) == 1, w >= 0, w[n] >= inp.target_cash_floor, cp.sum(w[:n]) <= inp.b_final, w[:n] <= inp.max_weights,
            0.5 * cp.sum(cp.abs(w - inp.w_minus)) <= inp.turnover_limit]
    if inp.usd_mask.any():
        cons.append(cp.sum(w[:n][inp.usd_mask]) <= inp.usd_limit)
    for i in np.flatnonzero(~inp.tradable):
        cons.append(w[i] == inp.w_minus[i])
    problem = cp.Problem(cp.Minimize(obj), cons)
    try:
        problem.solve(solver=cp.OSQP, **settings)
    except cp.error.SolverError:
        return SolveOutcome("solver_error", None, None, None, False, False, settings)
    info = getattr(getattr(problem.solver_stats, "extra_stats", None), "info", None)
    extra = getattr(problem.solver_stats, "extra_stats", None)
    return SolveOutcome(problem.status, None if w.value is None else np.asarray(w.value, dtype=float),
                        None if info is None else float(info.prim_res), None if info is None else float(info.dual_res),
                        getattr(extra, "prim_inf_cert", None) is not None and np.asarray(extra.prim_inf_cert).size > 0
                        and problem.status in ("infeasible", "infeasible_inaccurate"),
                        problem.status in ("unbounded", "unbounded_inaccurate"), settings)


def phase1_infeasible(inp: OptimizerInputs, settings: dict[str, Any]) -> bool:
    """Stricter zero-objective confirmation: the constraint set alone is infeasible."""
    flat = OptimizerInputs(inp.a_bar, inp.w_minus, inp.sigma, inp.w_track, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), inp.rates, inp.b_final,
                           inp.target_cash_floor, inp.max_weights, inp.usd_mask, inp.usd_limit, inp.turnover_limit, inp.tradable)
    out = solve_qp(flat, settings)
    return out.status in ("infeasible", "infeasible_inaccurate")


def reachable_min_gross(inp: OptimizerInputs) -> tuple[float, float, float]:
    """(minimum reachable risky gross, current gross, locked gross) under ordinary turnover (spec 8.4 analytical precheck)."""
    n = inp.n
    g_minus = float(inp.w_minus[:n].sum())
    g_locked = float(inp.w_minus[:n][~inp.tradable].sum())
    return max(g_locked, g_minus - inp.turnover_limit), g_minus, g_locked


def proportional_sells(inp: OptimizerInputs, target_gross: float) -> NDArray[np.float64]:
    """Deterministic sell-only de-risking: scale tradable risky positions down pro rata; buys are cancelled."""
    n = inp.n
    w = inp.w_minus.copy()
    locked = float(w[:n][~inp.tradable].sum())
    free = float(w[:n][inp.tradable].sum())
    goal = max(target_gross, locked)
    if free > 0 and goal < locked + free:
        w[:n][inp.tradable] *= (goal - locked) / free
    w[n] = 1.0 - w[:n].sum()
    return w


@dataclass(frozen=True)
class OptimizationResult:
    weights: NDArray[np.float64] | None
    branch: str
    status: str
    attempts: tuple[dict[str, Any], ...]
    terms: Terms | None
    turnover: float | None
    emergency: dict[str, Any] | None = None
    halted: bool = False
    halt_reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _terms(w: NDArray[np.float64], inp: OptimizerInputs) -> Terms:
    return objective_terms(w, inp.a_bar, inp.w_minus, np.asarray(inp.w_track), inp.sigma, inp.rates, inp.scales, inp.lambdas)


def _emergency(inp: OptimizerInputs, reason: str, attempts: list[dict]) -> OptimizationResult:
    """Sell-only de-risking that may exceed ordinary turnover by exactly the minimum needed (spec 8.4)."""
    _, g_minus, g_locked = reachable_min_gross(inp)
    goal = max(inp.b_final, g_locked)
    w = proportional_sells(inp, goal)
    required = g_minus - goal
    residual = float(max(0.0, g_locked - inp.b_final))
    info = {"reason": reason, "required_turnover": float(required), "applied_turnover": one_way_turnover(w, inp.w_minus),
            "ordinary_limit": inp.turnover_limit, "untradable_residual": residual, "goal_gross": float(goal)}
    if residual > 1e-12:  # `untradable_residual_policy: halt`
        return OptimizationResult(w, "emergency_sell_only", reason, tuple(attempts), _terms(w, inp), one_way_turnover(w, inp.w_minus),
                                  info, True, "untradable_residual", ("auditable ledger must be preserved before halting",))
    return OptimizationResult(w, "emergency_sell_only", reason, tuple(attempts), _terms(w, inp), one_way_turnover(w, inp.w_minus), info)


def cash_only(inp: OptimizerInputs, attempts: list[dict], reason: str) -> OptimizationResult:
    """Intention is zero risky / one cash; buys cancelled; deterministic sells of every tradable risky position."""
    n = inp.n
    w = inp.w_minus.copy()
    w[:n][inp.tradable] = 0.0
    w[n] = 1.0 - w[:n].sum()
    locked = float(w[:n].sum())
    info = {"reason": reason, "untradable_residual": locked, "ordinary_turnover_ignored": True}
    return OptimizationResult(w, "cash_only", "cash_only_pending" if locked == 0 else "cash_only_with_untradable_residual",
                              tuple(attempts), _terms(w, inp), one_way_turnover(w, inp.w_minus), info,
                              locked > 1e-12, "untradable_residual" if locked > 1e-12 else None)


def optimize(inp: OptimizerInputs, cfg: Config, *, previous_target: NDArray[np.float64] | None = None,
             first_decision: bool = False, solve: SolveFn = solve_qp) -> OptimizationResult:
    """Spec 8.4 failure-policy order, then the 8.5 status handling with a bounded number of attempts."""
    pol = cfg.solver_policy
    tol = cfg.reproducibility.tolerances.constraint_feasibility
    attempts: list[dict[str, Any]] = []

    min_reach, _, _ = reachable_min_gross(inp)
    if inp.b_final < min_reach - 1e-12:  # Defensive envelope outranks the ordinary turnover limit
        return _emergency(inp, "b_final_below_reachable_gross", attempts)

    settings = [{"eps_abs": pol.primary.eps_abs, "eps_rel": pol.primary.eps_rel, "max_iter": pol.primary.max_iter},
                {"eps_abs": pol.numerical_retry.eps_abs, "eps_rel": pol.numerical_retry.eps_rel,
                 "max_iter": pol.numerical_retry.max_iter, "polishing": pol.numerical_retry.polish}]
    limit = cfg.compute_budget.max_solver_attempts_per_problem
    for k, s in enumerate(settings[:limit]):
        out = solve(inp, s)
        viol = None if out.w is None else hard_constraint_violations(out.w, inp)
        ok_constraints = viol is not None and max(viol.values()) <= tol
        attempts.append({"attempt": k + 1, "status": out.status, "settings": s, "prim_res": out.prim_res, "dual_res": out.dual_res,
                         "max_violation": None if viol is None else max(viol.values()), "accepted": False})
        if out.dual_inf_cert or out.status in ("unbounded", "unbounded_inaccurate"):
            raise SpecificationError("dual infeasible: objective/specification error; no liquidation is attempted")
        if out.status in ("infeasible", "infeasible_inaccurate"):
            confirmed = out.prim_inf_cert and phase1_infeasible(inp, s)
            attempts[-1]["infeasibility_confirmed"] = bool(confirmed)
            if confirmed:
                return cash_only(inp, attempts, "confirmed_primal_infeasible")
            continue  # an unconfirmed infeasibility is treated as a numerical failure
        finite = out.w is not None and np.all(np.isfinite(out.w))
        if out.status == "optimal" and finite and ok_constraints:
            attempts[-1]["accepted"] = True
            return OptimizationResult(out.w, "optimal" if k == 0 else "numerical_retry_optimal", out.status, tuple(attempts),
                                      _terms(out.w, inp), one_way_turnover(out.w, inp.w_minus))
        if (out.status == "optimal_inaccurate" and finite and ok_constraints and out.prim_res is not None
                and out.prim_res <= pol.acceptance.primal_residual_max and out.dual_res is not None
                and out.dual_res <= pol.acceptance.dual_residual_max):
            attempts[-1]["accepted"] = True
            return OptimizationResult(out.w, "optimal_inaccurate_accepted", out.status, tuple(attempts), _terms(out.w, inp),
                                      one_way_turnover(out.w, inp.w_minus))

    # fallback_chain after a failed retry: hold previous -> marked holdings -> emergency sell-only -> halt (spec 8.5 item 4)
    no_turnover = 1e9  # holding a prior intention or the marked state is not itself a rebalance
    if previous_target is not None:
        pv = hard_constraint_violations(previous_target, inp, no_turnover)
        if pv["risky_gross"] <= tol and max(v for k, v in pv.items() if k != "risky_gross") <= tol:
            notes = ("previous target requires more than the ordinary one-way turnover",)                 if one_way_turnover(previous_target, inp.w_minus) > inp.turnover_limit + tol else ()
            return OptimizationResult(previous_target.copy(), "hold_previous", "fallback", tuple(attempts),
                                      _terms(previous_target, inp), one_way_turnover(previous_target, inp.w_minus), notes=notes)
    mv = hard_constraint_violations(inp.w_minus, inp, no_turnover)
    if max(mv.values()) <= tol:  # first decision, or the previous target exceeds B_final: do not trade
        return OptimizationResult(inp.w_minus.copy(), "use_marked_holdings", "fallback", tuple(attempts), _terms(inp.w_minus, inp), 0.0)
    if mv["risky_gross"] > tol and max(v for k, v in mv.items() if k != "risky_gross") <= tol:
        return _emergency(inp, "gross_exceeds_b_final_after_solver_failure", attempts)
    return OptimizationResult(None, "halt", "no_self_financing_path", tuple(attempts), None, None, None, True, "no_self_financing_path",
                              tuple(f"marked holdings violate {k}" for k, v in mv.items() if v > tol))
