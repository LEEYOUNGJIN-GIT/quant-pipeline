"""Diversification agent: long-only minimum variance over a risky sleeve with runtime eligible-set clipping (spec 8.2).

The optimization variable ``q`` is a risky-sleeve share (q >= 0, sum q = 1), not an absolute weight. It does not use
downside, drawdown or the Defensive cap.
"""
from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from ..config import Config
from .base import CASH, AgentContext, AgentOutput, build_output

TOL = 1e-8
STATS = {"problems": 0, "attempts": 0}  # auxiliary-solve accounting (spec 11.8): problems and solver attempts


def capped_simplex(weights: NDArray[np.float64], caps: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Deterministic projection: proportional shares, capped assets fixed at their cap, remainder redistributed."""
    if caps.sum() < 1.0 - 1e-12:
        return None
    q = np.zeros(len(caps))
    free = np.ones(len(caps), dtype=bool)
    remaining = 1.0
    for _ in range(len(caps) + 1):
        share = weights[free] / weights[free].sum() * remaining
        over = share > caps[free] + 1e-15
        if not over.any():
            q[free] = share
            return q
        idx = np.flatnonzero(free)[over]
        q[idx] = caps[idx]
        remaining -= caps[idx].sum()
        free[idx] = False
        if not free.any():
            return q if abs(q.sum() - 1.0) < 1e-9 else None
    return None


def repair_to_caps(q: NDArray[np.float64], caps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove solver-tolerance violations: clip into [0, caps] then hand the leftover mass to assets below their cap."""
    q = np.clip(q, 0.0, caps)
    for _ in range(len(q) + 1):
        gap = 1.0 - q.sum()
        room = caps - q
        if abs(gap) < 1e-15 or room.sum() <= 0:
            break
        q = np.clip(q + gap * room / room.sum(), 0.0, caps) if gap > 0 else np.clip(q + gap * q / q.sum(), 0.0, caps)
    return q


def _feasible(q: NDArray[np.float64], caps: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(q)) and abs(q.sum() - 1.0) <= TOL and np.all(q >= -TOL) and np.all(q <= caps + TOL))


def solve_minvar(sigma: NDArray[np.float64], caps: NDArray[np.float64], cfg: Config) -> tuple[NDArray[np.float64] | None, dict[str, Any]]:
    """Primary OSQP solve, one numerical retry (2 attempts total, spec 8.5), each independently feasibility-checked."""
    sp = cfg.solver_policy
    info: dict[str, Any] = {"attempts": []}
    STATS["problems"] += 1
    attempts = [{"eps_abs": sp.primary.eps_abs, "eps_rel": sp.primary.eps_rel, "max_iter": sp.primary.max_iter},
                {"eps_abs": sp.numerical_retry.eps_abs, "eps_rel": sp.numerical_retry.eps_rel,
                 "max_iter": sp.numerical_retry.max_iter, "polishing": sp.numerical_retry.polish}]  # osqp>=1.0 name for `polish`
    for opts in attempts:
        STATS["attempts"] += 1
        q = cp.Variable(len(caps))
        problem = cp.Problem(cp.Minimize(cp.quad_form(q, cp.psd_wrap(sigma))), [q >= 0, cp.sum(q) == 1, q <= caps])
        try:
            problem.solve(solver=cp.OSQP, **opts)
            value = None if q.value is None else np.asarray(q.value, dtype=float)
            ok = value is not None and problem.status in ("optimal", "optimal_inaccurate") and _feasible(value, caps)
            info["attempts"].append({"status": problem.status, "settings": opts, "accepted": bool(ok)})
            if ok:
                return repair_to_caps(value, caps), info
        except cp.error.SolverError as e:
            info["attempts"].append({"status": "solver_error", "settings": opts, "accepted": False, "error": str(e)})
    return None, info


def run(ctx: AgentContext, cfg: Config) -> AgentOutput:
    f, a = cfg.features, cfg.agents.diversification
    n = len(ctx.assets)
    eligible = ctx.tradable.copy()
    coverage = float(eligible.mean())
    cap_sum = float(ctx.max_weights[eligible].sum())
    proposed = a.risk_budget_cap * coverage
    b_eff = min(proposed, cap_sum)
    diag: dict[str, Any] = {"B_div_proposed": proposed, "eligible_cap_sum": cap_sum, "B_div_effective": b_eff,
                            "eligible": eligible.tolist(), "coverage": coverage}

    if b_eff < a.min_meaningful_risky_budget or not eligible.any():
        diag.update(agent_status="degraded_no_meaningful_budget",
                    reason="B_div_effective below min_meaningful_risky_budget or empty eligible set")
        out = build_output(fold_id=ctx.fold_id, refit_id=ctx.refit_id, as_of=ctx.as_of, name="Diversification",
                           assets=ctx.assets, raw=np.zeros(n), eligible=np.zeros(n, dtype=bool), confidence=0.0,
                           risk_budget_cap=a.risk_budget_cap, score_mass_floor=f.score_mass_floor, diagnostics=diag)
        return out

    idx = np.flatnonzero(eligible)
    sigma = ctx.covariance[np.ix_(idx, idx)]
    lam_min = float(np.linalg.eigvalsh(sigma)[0])
    rho = max(a.minvar_ridge, -lam_min + a.minvar_eigen_floor)
    caps = np.minimum(1.0, ctx.max_weights[idx] / b_eff)
    diag.update(ridge=rho, min_eigenvalue=lam_min, sleeve_caps=caps.tolist())

    q_sub = None
    if caps.sum() >= 1.0 - 1e-12:
        q_sub, info = solve_minvar(sigma + rho * np.eye(len(idx)), caps, cfg)
        diag["solver"] = info
        diag["method"] = "long_only_minimum_variance" if q_sub is not None else None
        if q_sub is None:  # declared fallback: inverse volatility followed by deterministic capped-simplex projection
            vol = np.maximum(np.sqrt(np.diag(sigma)), a.volatility_floor)
            q_sub = capped_simplex(1.0 / vol, caps)
            diag["method"] = "inverse_volatility_capped_simplex" if q_sub is not None else None
            diag["fallback_used"] = True
    if q_sub is None:
        diag.update(agent_status="degraded_no_meaningful_budget", reason="sleeve caps infeasible or no valid solution")
        return build_output(fold_id=ctx.fold_id, refit_id=ctx.refit_id, as_of=ctx.as_of, name="Diversification",
                            assets=ctx.assets, raw=np.zeros(n), eligible=np.zeros(n, dtype=bool), confidence=0.0,
                            risk_budget_cap=a.risk_budget_cap, score_mass_floor=f.score_mass_floor, diagnostics=diag)

    q = np.zeros(n)
    q[idx] = q_sub
    diag["agent_status"] = "ok"
    diag["residuals"] = {"sum_error": float(abs(q.sum() - 1.0)), "cap_violation": float(max(0.0, (q[idx] - caps).max()))}
    # build the output on the sleeve shares, then enforce the effective (clipped) budget explicitly
    out = build_output(fold_id=ctx.fold_id, refit_id=ctx.refit_id, as_of=ctx.as_of, name="Diversification",
                       assets=ctx.assets, raw=q, eligible=eligible, confidence=1.0, risk_budget_cap=1.0,
                       score_mass_floor=f.score_mass_floor, diagnostics=diag)
    target = {k: min(b_eff * v, float(ctx.max_weights[i])) for i, (k, v) in enumerate(out.normalized_score.items())}
    target[CASH] = 1.0 - sum(target.values())  # the cap is rechecked in portfolio-weight units; any excess stays in cash
    return AgentOutput(out.fold_id, out.refit_id, out.as_of_utc, "Diversification", out.raw_score, out.normalized_score,
                       b_eff / a.risk_budget_cap if a.risk_budget_cap else 0.0, b_eff, target, 0.0, out.diagnostics)
