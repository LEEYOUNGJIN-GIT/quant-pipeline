"""Ledoit-Wolf covariance on the canonical synchronized panel (spec 7.4)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf, empirical_covariance

JSONValue = Any


class PsdRepairError(ValueError):
    """A numerical PSD repair exceeded the configured Frobenius threshold: the baseline fails."""


@dataclass(frozen=True)
class CovarianceResult:
    fold_id: str
    refit_id: str
    asset_ids: tuple[str, ...]
    matrix: NDArray[np.float64]
    estimator_name: str
    fit_start_utc: datetime
    fit_end_utc: datetime
    n_observations: int
    effective_observations: Mapping[str, int]
    diagnostics: Mapping[str, JSONValue]


def repair_psd(matrix: NDArray[np.float64], eigen_floor: float, max_frobenius: float) -> tuple[NDArray[np.float64], dict]:
    """Clip eigenvalues below ``eigen_floor`` to zero only if needed; fail if the change is too large."""
    sym = (matrix + matrix.T) / 2.0
    w, v = np.linalg.eigh(sym)
    info: dict[str, Any] = {"eigenvalues_before": w.tolist(), "repaired": False, "frobenius_repair": 0.0}
    if w[0] >= eigen_floor:
        return sym, info
    fixed = (v * np.clip(w, 0.0, None)) @ v.T
    fixed = (fixed + fixed.T) / 2.0
    mag = float(np.linalg.norm(fixed - sym, "fro"))
    info.update(repaired=True, frobenius_repair=mag, eigenvalues_after=np.linalg.eigvalsh(fixed).tolist())
    if mag > max_frobenius:
        raise PsdRepairError(f"PSD repair {mag:.3e} exceeds threshold {max_frobenius:.3e}")
    return fixed, info


def fit_ledoit_wolf(returns: NDArray[np.float64], asset_ids: tuple[str, ...], *, fold_id: str, refit_id: str,
                    fit_start_utc: datetime, fit_end_utc: datetime, effective_observations: Mapping[str, int],
                    min_observations: int, eigen_floor: float, max_frobenius: float) -> CovarianceResult:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(asset_ids):
        raise ValueError("returns must be (rows, assets) in canonical asset order")
    if not np.all(np.isfinite(x)):
        raise ValueError("Ledoit-Wolf input must be finite complete-case rows")
    if x.shape[0] < min_observations:
        raise ValueError(f"{x.shape[0]} rows < minimum {min_observations}")
    model = LedoitWolf(assume_centered=False).fit(x)
    matrix, psd = repair_psd(model.covariance_, eigen_floor, max_frobenius)
    eig = np.linalg.eigvalsh(matrix)
    corr_scale = np.sqrt(np.diag(matrix))
    corr = matrix / np.outer(corr_scale, corr_scale)
    diagnostics = {
        "shrinkage": float(model.shrinkage_),
        "location": model.location_.tolist(),
        "sample_covariance": empirical_covariance(x, assume_centered=False).tolist(),  # estimator convention (1/n)
        "shrunk_covariance": model.covariance_.tolist(),
        "eigenvalues": eig.tolist(),
        "condition_number": float(eig[-1] / eig[0]) if eig[0] > 0 else None,
        "psd": psd,
        "correlation": corr.tolist(),
    }
    return CovarianceResult(fold_id, refit_id, asset_ids, matrix, "LedoitWolf(assume_centered=False)", fit_start_utc,
                            fit_end_utc, int(x.shape[0]), dict(effective_observations), diagnostics)
