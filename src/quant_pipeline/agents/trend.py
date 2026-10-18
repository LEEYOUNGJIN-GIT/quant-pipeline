"""Trend agent: weighted lagged total returns divided by trailing volatility, clipped (spec 8.2)."""
from __future__ import annotations

import bisect

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from ..risk_pressure import type7_quantile
from .base import AgentContext, AgentOutput, History, build_output


def scores_at(returns: NDArray[np.float64], t: int, horizons: list[int], weights: list[float], vol_window: int,
              clip_z: float, min_obs: int) -> tuple[NDArray[np.float64], float]:
    """Raw scores and horizon coverage at row ``t`` using rows <= t only."""
    n_assets = returns.shape[1]
    have = t + 1
    vol_rows = returns[max(0, have - vol_window): have]
    if vol_rows.shape[0] < min_obs:
        return np.zeros(n_assets), 0.0
    vol = np.maximum(vol_rows.std(axis=0, ddof=1), 1e-8)
    total, covered = np.zeros(n_assets), 0.0
    for h, w in zip(horizons, weights, strict=True):
        if have >= h:
            total += w * (np.prod(1.0 + returns[have - h: have], axis=0) - 1.0)
            covered += w
    coverage = covered / float(sum(weights))
    return np.clip(total / vol, -clip_z, clip_z), coverage


def _masses(returns: NDArray[np.float64], cfg: Config, weights: list[float], start: int, end: int) -> tuple[list[int], list[float]]:
    """(rows, positive score mass) for every row in ``start..end`` that has full horizon coverage and a positive mass."""
    f = cfg.features
    rows, masses = [], []
    for t in range(start, end + 1):
        s, cov = scores_at(returns, t, f.return_horizons, weights, f.volatility_window, f.clip_z, f.min_observations)
        mass = float(np.clip(s, 0.0, None).sum())
        if cov >= 1.0 and mass > 0.0:
            rows.append(t)
            masses.append(mass)
    return rows, masses


def positive_mass_reference(returns: NDArray[np.float64], cfg: Config, weights: list[float], start: int, end: int,
                            history: History | None = None) -> float:
    """Type-7 median of strictly positive score mass over training-only decisions, floored by ``score_mass_floor``.

    The mass at row ``t`` uses rows ``<= t`` only, so with a ``history`` the masses are computed once over the whole history and the ones up to
    ``end`` are taken: exactly the list a fresh pass over ``returns`` would build.
    """
    f = cfg.features
    if history is not None and history.is_prefix(returns) and end < history.returns.shape[0]:
        key = ("trend_mass", start, tuple(weights), tuple(f.return_horizons), f.volatility_window, f.clip_z, f.min_observations)
        if key not in history.memo:
            history.memo[key] = _masses(history.returns, cfg, weights, start, history.returns.shape[0] - 1)
        rows, all_masses = history.memo[key]
        masses = all_masses[: bisect.bisect_right(rows, end)]
    else:
        masses = _masses(returns, cfg, weights, start, end)[1]
    return max(type7_quantile(np.array(masses), 0.5), f.score_mass_floor) if masses else f.score_mass_floor


def run(ctx: AgentContext, cfg: Config, horizon_weights: list[float] | None = None) -> AgentOutput:
    f, a = cfg.features, cfg.agents.trend
    w = horizon_weights or a.horizon_weight_candidates[0]
    t = ctx.returns.shape[0] - 1
    raw, coverage = scores_at(ctx.returns, t, f.return_horizons, w, f.volatility_window, f.clip_z, f.min_observations)
    ref = positive_mass_reference(ctx.returns, cfg, w, ctx.train_start_row, ctx.train_end_row, ctx.history)
    mass = float(np.clip(np.where(ctx.tradable, raw, 0.0), 0.0, None).sum())
    confidence = coverage * min(1.0, mass / ref)
    return build_output(fold_id=ctx.fold_id, refit_id=ctx.refit_id, as_of=ctx.as_of, name="Trend", assets=ctx.assets,
                        raw=raw, eligible=ctx.tradable & (coverage > 0), confidence=confidence,
                        risk_budget_cap=a.risk_budget_cap, score_mass_floor=f.score_mass_floor,
                        diagnostics={"horizon_coverage": coverage, "positive_mass": mass, "reference_positive_mass": ref,
                                     "horizon_weights": list(w), "calibration_rows": [ctx.train_start_row, ctx.train_end_row]})
