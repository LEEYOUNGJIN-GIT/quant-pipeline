"""MeanReversion agent: negative clipped PCA-residual z-score with an entry/exit state machine (spec 8.2)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from .base import AgentContext, AgentOutput, build_output


def residual_matrix(ctx: AgentContext) -> NDArray[np.float64]:
    """Residuals of every history row against the *frozen* fold PCA (standardized units)."""
    z = (ctx.returns - ctx.standardizer_mean) / ctx.standardizer_scale
    return z - (z @ ctx.loadings.T) @ ctx.loadings


def z_series(resid: NDArray[np.float64], window: int, min_obs: int) -> NDArray[np.float64]:
    """Rolling z-score of the residual using the trailing window ending at each row; NaN until ``min_obs`` rows exist."""
    n, k = resid.shape
    z = np.full((n, k), np.nan)
    for t in range(n):
        rows = resid[max(0, t + 1 - window): t + 1]
        if rows.shape[0] >= min_obs:
            sd = np.maximum(rows.std(axis=0, ddof=1), 1e-12)
            z[t] = (resid[t] - rows.mean(axis=0)) / sd
    return z


def replay_state(z: NDArray[np.float64], entry_z: float, exit_z: float) -> NDArray[np.bool_]:
    """Long reversion trade opens when z <= -entry and closes when z >= -exit; replayed deterministically from history."""
    active = np.zeros(z.shape[1], dtype=bool)
    for t in range(z.shape[0]):
        row = z[t]
        if np.any(np.isnan(row)):
            continue
        active = np.where(active, ~(row >= -exit_z), row <= -entry_z)
    return active


def run(ctx: AgentContext, cfg: Config, entry_z: float | None = None) -> AgentOutput:
    f, a = cfg.features, cfg.agents.mean_reversion
    entry = entry_z if entry_z is not None else a.entry_z_candidates[0]
    z = z_series(residual_matrix(ctx), f.residual_z_window, f.min_observations)
    zt = z[-1]
    valid = ~np.isnan(zt)
    active = replay_state(z, entry, a.exit_z) & ctx.tradable & valid
    raw = np.where(active, -np.clip(np.nan_to_num(zt), -f.clip_z, f.clip_z), 0.0)
    coverage = float(valid.mean())
    eligible = ctx.tradable & valid
    fraction = float(active.sum()) / max(int(eligible.sum()), 1)
    return build_output(fold_id=ctx.fold_id, refit_id=ctx.refit_id, as_of=ctx.as_of, name="MeanReversion", assets=ctx.assets,
                        raw=raw, eligible=eligible, confidence=fraction * coverage, risk_budget_cap=a.risk_budget_cap,
                        score_mass_floor=f.score_mass_floor,
                        diagnostics={"z": [None if np.isnan(v) else float(v) for v in zt], "active": active.tolist(),
                                     "entry_z": entry, "exit_z": a.exit_z, "data_coverage": coverage,
                                     "active_entry_fraction": fraction})
