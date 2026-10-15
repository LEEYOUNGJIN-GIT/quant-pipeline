"""Canonical objective math and train-only calibration of ``W_track`` and the scales (spec 9.1-9.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .risk_pressure import type7_quantile


def expected_cost_rates(cfg: Config, multiplier: float = 1.0) -> NDArray[np.float64]:
    """Per-unit |dw| rate for each risky asset then cash (rate 0). Taxes: half of the sell-only tax (OQ-019)."""
    c = cfg.costs
    rates = []
    for a in cfg.universe.assets or []:
        bps = c.commission_bps + c.half_spread_slippage_bps + (c.fx_conversion_bps if a.currency == "USD" else 0.0)
        bps += 0.5 * c.sell_tax_bps[a.exchange]
        rates.append(bps / 1e4)
    return np.array([*rates, 0.0]) * multiplier


def one_way_turnover(w: NDArray[np.float64], w_minus: NDArray[np.float64]) -> float:
    return float(0.5 * np.abs(w - w_minus).sum())


def drift(weights_plus: NDArray[np.float64], risky_returns: NDArray[np.float64], cash_return: float = 0.0) -> NDArray[np.float64]:
    """Drift post-trade full weights by realized returns, then renormalize the full vector (spec 9.2 step 5)."""
    grown = weights_plus * np.append(1.0 + np.asarray(risky_returns, dtype=float), 1.0 + cash_return)
    return grown / grown.sum()


@dataclass(frozen=True)
class Terms:
    track: float
    risk: float
    cost: float
    objective: float
    contributions: tuple[float, float, float]


def objective_terms(w: NDArray[np.float64], a_bar: NDArray[np.float64], w_minus: NDArray[np.float64], w_track: NDArray[np.float64],
                    sigma: NDArray[np.float64], rates: NDArray[np.float64], scales: tuple[float, float, float],
                    lambdas: tuple[float, float, float]) -> Terms:
    """The three normalized terms of spec 9.1 evaluated independently of any solver."""
    n = len(sigma)
    d = w - a_bar
    track = float(d @ np.diag(w_track) @ d)
    risk = float(w[:n] @ sigma @ w[:n])
    cost = float(rates @ np.abs(w - w_minus))
    parts = (lambdas[0] * track / scales[0], lambdas[1] * risk / scales[1], lambdas[2] * cost / scales[2])
    return Terms(track, risk, cost, float(sum(parts)), parts)


def w_track_diagonal(deltas: NDArray[np.float64], floor: float, w_min: float, w_max: float) -> NDArray[np.float64]:
    """omega_i = (N+1) u_i / sum_j u_j with u_i = clip(1/d_i, w_min, w_max), d_i = max(Type7Median(delta_i^2), floor)."""
    d = np.maximum([type7_quantile(deltas[:, i] ** 2, 0.5) for i in range(deltas.shape[1])], floor)
    u = np.clip(1.0 / d, w_min, w_max)
    return deltas.shape[1] * u / u.sum()


@dataclass(frozen=True)
class Scale:
    name: str
    value: float
    fallback: bool
    sorted_sample: tuple[float, ...]
    floor: float
    n_positive: int


def robust_scale(name: str, values: NDArray[np.float64], floor: float) -> Scale:
    """``max(Type7Median(strictly positive finite q), floor)``; no positive value -> floor with ``scale_fallback``."""
    x = np.sort(np.asarray(values, dtype=float))
    pos = x[np.isfinite(x) & (x > 0)]
    if len(pos) == 0:
        return Scale(name, floor, True, (), floor, 0)
    return Scale(name, max(type7_quantile(pos, 0.5), floor), False, tuple(pos.tolist()), floor, len(pos))


@dataclass(frozen=True)
class ObjectiveCalibration:
    w_track: NDArray[np.float64]
    s_track: Scale
    s_risk: Scale
    s_cost: Scale
    n_decisions: int
    excluded: tuple[tuple[int, str], ...]
    v_minus: NDArray[np.float64]
    a_bar: NDArray[np.float64]
    diagnostics: dict[str, Any]

    @property
    def scales(self) -> tuple[float, float, float]:
        return (self.s_track.value, self.s_risk.value, self.s_cost.value)


def calibrate_objective(cfg: Config, a_bar: NDArray[np.float64], v_minus: NDArray[np.float64],
                        sigmas: list[NDArray[np.float64] | None]) -> ObjectiveCalibration:
    """Calibrate from the virtual pretrade rows. ``sigmas[j] is None`` excludes row j from the risk scale (with reason)."""
    c = cfg.objective.calibration
    deltas = a_bar - v_minus
    w_track = w_track_diagonal(deltas, c.component_scale_floor, c.diagonal_weight_min, c.diagonal_weight_max)
    rates = expected_cost_rates(cfg, 1.0)  # S_cost is always calibrated at 1x
    n = a_bar.shape[1] - 1
    q_track = np.array([d @ np.diag(w_track) @ d for d in deltas])
    q_cost = np.abs(deltas) @ rates
    q_risk, excluded = [], []
    for j, s in enumerate(sigmas):
        if s is None:
            q_risk.append(np.nan)
            excluded.append((j, "insufficient_covariance_history"))
        else:
            q_risk.append(float(a_bar[j, :n] @ s @ a_bar[j, :n]))
    return ObjectiveCalibration(
        w_track, robust_scale("S_track", q_track, c.scale_floor_track), robust_scale("S_risk", np.array(q_risk), c.scale_floor_risk),
        robust_scale("S_cost", q_cost, c.scale_floor_cost), a_bar.shape[0], tuple(excluded), v_minus, a_bar,
        {"trace_w_track": float(w_track.sum()), "component_scale_floor": c.component_scale_floor})
