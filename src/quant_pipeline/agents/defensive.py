"""Defensive agent: downside and common-factor fragility (spec 8.2). Not the Diversification formula."""
from __future__ import annotations

import numpy as np

from ..config import Config
from .base import AgentContext, AgentOutput, build_output


def run(ctx: AgentContext, cfg: Config) -> AgentOutput:
    f, a = cfg.features, cfg.agents.defensive
    r = ctx.returns
    n = r.shape[0]
    down_rows = r[max(0, n - a.downside_window):]
    valid = np.full(len(ctx.assets), down_rows.shape[0] >= f.min_observations)
    downside = np.sqrt(np.mean(np.minimum(down_rows, 0.0) ** 2, axis=0))
    loading = np.abs(ctx.loadings[0])
    wealth = np.cumprod(1.0 + r[max(0, n - f.drawdown_window):], axis=0)
    drawdown = np.abs(np.minimum(wealth[-1] / wealth.max(axis=0) - 1.0, 0.0))
    liquidity = ctx.tradable.astype(float)
    raw = (liquidity / np.maximum(downside, a.downside_floor)
           * np.exp(-a.factor_loading_penalty * loading) * np.exp(-a.drawdown_penalty * drawdown))
    eligible = ctx.tradable & valid
    coverage = float(eligible.mean())
    return build_output(fold_id=ctx.fold_id, refit_id=ctx.refit_id, as_of=ctx.as_of, name="Defensive", assets=ctx.assets,
                        raw=raw, eligible=eligible, confidence=coverage, risk_budget_cap=a.risk_budget_cap,
                        score_mass_floor=f.score_mass_floor,
                        diagnostics={"downside": downside.tolist(), "first_pca_abs_loading": loading.tolist(),
                                     "trailing_drawdown_magnitude": drawdown.tolist(), "valid_data_coverage": coverage})
