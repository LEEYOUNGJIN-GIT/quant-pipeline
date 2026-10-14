"""Stationary block bootstrap indices (spec 11.7) over the continuous aligned outer-OOS valuation returns.

Algorithm ``oracle-stationary-bootstrap-v1-politis-romano`` (OQ-007): per replicate the first index is uniform; each next
step starts a fresh block with probability ``1 / mean_block_length`` (a new uniform index) and otherwise continues at
``(previous + 1) mod n``. One ``Generator(PCG64(seed))`` is consumed row-major.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ALGORITHM_ID = "oracle-stationary-bootstrap-v1-politis-romano"


def index_matrix(n: int, replicates: int, mean_block_length: int, seed: int) -> NDArray[np.int64]:
    if n < 2 or replicates < 1 or mean_block_length < 1:
        raise ValueError("invalid bootstrap dimensions")
    rng = np.random.Generator(np.random.PCG64(seed))
    p = 1.0 / mean_block_length
    m = np.empty((replicates, n), dtype="<i8")
    for r in range(replicates):
        m[r, 0] = rng.integers(0, n)
        for t in range(1, n):
            m[r, t] = rng.integers(0, n) if rng.random() < p else (m[r, t - 1] + 1) % n
    return m


def index_hash(m: NDArray[np.int64]) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(m, dtype="<i8").tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class BootstrapResult:
    indices: NDArray[np.int64]
    index_hash: str
    statistics: dict[str, dict[str, float]]
    inferential: bool
    status: str


def percentile_interval(samples: NDArray[np.float64], confidence: float) -> tuple[float, float]:
    a = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, a, method="linear")), float(np.quantile(samples, 1.0 - a, method="linear"))


def run_bootstrap(returns: dict[str, NDArray[np.float64]], rf: NDArray[np.float64], *, replicates: int, mean_block_length: int,
                  seed: int, confidence: float, periods_per_year: int, minimum_rows: int, expected_rows: int | None,
                  inferential: bool) -> BootstrapResult:
    """Resample the *aligned tuple* of series with one shared index matrix (series are never resampled independently)."""
    n = len(rf)
    if any(len(v) != n for v in returns.values()):
        raise ValueError("bootstrap series must be aligned to the same valuation intervals")
    if expected_rows is not None and n != expected_rows:
        raise ValueError(f"bootstrap input has {n} intervals, expected {expected_rows}")
    if n < minimum_rows:
        return BootstrapResult(np.empty((0, 0), dtype="<i8"), "", {}, inferential, "insufficient_sample")
    idx = index_matrix(n, replicates, mean_block_length, seed)
    stats: dict[str, dict[str, float]] = {}
    for name, r in returns.items():
        ex = np.asarray(r, dtype=float) - rf
        sampled = ex[idx]
        sd = sampled.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, sampled.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(periods_per_year), np.nan)
        finite = sharpe[np.isfinite(sharpe)]
        point_sd = ex.std(ddof=1)
        point = float(ex.mean() / point_sd * np.sqrt(periods_per_year)) if point_sd > 0 else float("nan")
        if finite.size == 0:  # e.g. the cash reference minus its own risk-free leg is identically zero: undefined, never coerced to 0
            stats[name] = {"point_sharpe": point, "ci_low": float("nan"), "ci_high": float("nan"), "replicates": 0.0,
                           "undefined": 1.0}
            continue
        lo, hi = percentile_interval(finite, confidence)
        stats[name] = {"point_sharpe": point, "ci_low": lo, "ci_high": hi, "replicates": float(len(finite)), "undefined": 0.0}
    return BootstrapResult(idx, index_hash(idx), stats, inferential, "ok")
