"""Train-only risk-pressure thresholds (spec 8.3). Observed pressure at a decision is compared with thresholds fitted from
training decisions only; OOS and holdout values never recalibrate them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .factors import rolling_first_pc_share

METRICS = ("market_drawdown_magnitude", "first_pca_variance_share", "illiquid_fraction")


def type7_quantile(values: NDArray[np.float64], q: float) -> float:
    """Hyndman-Fan type 7 (numpy's default linear interpolation)."""
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def pressure(x: float, start: float, full: float, gap_floor: float) -> float:
    return float(np.clip((x - start) / max(full - start, gap_floor), 0.0, 1.0))


@dataclass(frozen=True)
class Threshold:
    metric: str
    start: float
    full: float
    fallback: bool
    n_training: int
    reason: str


def calibrate(metric: str, training: NDArray[np.float64], *, start_q: float, full_q: float, gap_floor: float,
              min_decisions: int, fallback: tuple[float, float]) -> Threshold:
    """Fit thresholds from training values only; use the configured training-only fallback if sample or spread is short."""
    x = np.asarray(training, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < min_decisions:
        return Threshold(metric, fallback[0], fallback[1], True, len(x), "insufficient_training_decisions")
    lo, hi = type7_quantile(x, start_q), type7_quantile(x, full_q)
    if hi - lo < gap_floor:
        return Threshold(metric, fallback[0], fallback[1], True, len(x), "insufficient_spread")
    return Threshold(metric, lo, hi, False, len(x), "fitted")


def market_drawdown_magnitude(returns: NDArray[np.float64], window: int) -> float:
    """Drawdown magnitude of the equal-weight portfolio over the trailing window (causal)."""
    r = np.asarray(returns, dtype=float)[-window:].mean(axis=1)
    wealth = np.cumprod(1.0 + r)
    return float(1.0 - wealth[-1] / wealth.max())


def metric_series(returns: NDArray[np.float64], illiquid: NDArray[np.float64], *, drawdown_window: int, factor_window: int,
                  factor_min_obs: int, first_row: int) -> dict[str, NDArray[np.float64]]:
    """Each metric at every row u >= first_row using only rows <= u (a fold-end value is never copied backward)."""
    n = returns.shape[0]
    out = {m: np.full(n, np.nan) for m in METRICS}
    for u in range(first_row, n):
        out["market_drawdown_magnitude"][u] = market_drawdown_magnitude(returns[: u + 1], drawdown_window)
        share = rolling_first_pc_share(returns[max(0, u + 1 - factor_window): u + 1], factor_min_obs)
        out["first_pca_variance_share"][u] = np.nan if share is None else share
        out["illiquid_fraction"][u] = illiquid[u]
    return out
