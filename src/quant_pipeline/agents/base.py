"""Common agent contract (spec 8.1): every agent returns all risky assets in canonical order plus CASH_KRW."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

AgentName = Literal["Trend", "MeanReversion", "Defensive", "Diversification"]
AGENTS: tuple[AgentName, ...] = ("Trend", "MeanReversion", "Defensive", "Diversification")
CASH = "CASH_KRW"


@dataclass(frozen=True)
class AgentOutput:
    fold_id: str
    refit_id: str
    as_of_utc: datetime
    agent_name: AgentName
    raw_score: Mapping[str, float]
    normalized_score: Mapping[str, float]
    confidence: float
    proposed_risky_budget: float
    absolute_target: Mapping[str, float]
    regime_gate: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def vector(self, assets: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([self.absolute_target[a] for a in (*assets, CASH)])

    def with_gate(self, gate: float) -> AgentOutput:
        return replace(self, regime_gate=gate, diagnostics={**self.diagnostics, "gate_applied": True})


def transform_scores(raw: NDArray[np.float64], eligible: NDArray[np.bool_], confidence: float, risk_budget_cap: float,
                     score_mass_floor: float) -> tuple[NDArray[np.float64], float, float]:
    """The identical long-only transformation for every agent (spec 8.1).

    Returns (normalized_score, proposed_risky_budget, clipped_confidence). A negative score means no long allocation; it is
    never shorted, abs()-ed, soft-maxed or renormalized into other assets.
    """
    raw = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw scores must be finite")
    positive = np.where(eligible, np.clip(raw, 0.0, None), 0.0)
    mass = float(positive.sum())
    conf = float(np.clip(confidence, 0.0, 1.0))
    if mass > score_mass_floor:
        return positive / mass, risk_budget_cap * conf, conf
    return np.zeros_like(positive), 0.0, conf


def build_output(*, fold_id: str, refit_id: str, as_of: datetime, name: AgentName, assets: tuple[str, ...],
                 raw: NDArray[np.float64], eligible: NDArray[np.bool_], confidence: float, risk_budget_cap: float,
                 score_mass_floor: float, diagnostics: Mapping[str, Any]) -> AgentOutput:
    norm, budget, conf = transform_scores(raw, eligible, confidence, risk_budget_cap, score_mass_floor)
    target = {a: float(budget * n) for a, n in zip(assets, norm, strict=True)}
    target[CASH] = float(1.0 - budget)
    return AgentOutput(fold_id, refit_id, as_of, name, {a: float(r) for a, r in zip(assets, raw, strict=True)},
                       {a: float(n) for a, n in zip(assets, norm, strict=True)}, conf, float(budget), target, 0.0,
                       {**diagnostics, "gate_applied": False, "agent_status": diagnostics.get("agent_status", "ok")})


@dataclass(frozen=True)
class AgentContext:
    """Everything an agent may read at one decision: causal history up to the cutoff plus the frozen fold fit."""

    fold_id: str
    refit_id: str
    as_of: datetime
    assets: tuple[str, ...]
    returns: NDArray[np.float64]  # complete-case simple returns up to and including the decision row, canonical order
    tradable: NDArray[np.bool_]
    loadings: NDArray[np.float64]  # frozen first-PC loadings (standardized units)
    standardizer_mean: NDArray[np.float64]
    standardizer_scale: NDArray[np.float64]
    covariance: NDArray[np.float64]
    max_weights: NDArray[np.float64]
    train_end_row: int  # last row index (inclusive) that belongs to the training window for calibration
    train_start_row: int  # first training-decision row index
    illiquid_history: NDArray[np.float64] | None = None  # fraction of assets not tradable at each history row
    row_cutoffs: tuple[str, ...] = ()  # decision cutoff (UTC) of each complete-case history row
