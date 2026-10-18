"""Fold-local standardization, PCA and the causal rolling PCA pressure statistic (spec 7.2, 7.3, 8.3).

Nothing here sees data beyond the rows it is given; callers pass only rows permitted by the fit cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

SCALE_FLOOR = 1e-12  # absolute; an asset with a smaller sample std is invalid (OQ-014)


class InsufficientObservations(ValueError):
    """Fewer permitted rows than the configured minimum."""


@dataclass(frozen=True)
class Standardizer:
    asset_ids: tuple[str, ...]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    clip_z: float
    n_observations: int

    @classmethod
    def fit(cls, returns: NDArray[np.float64], asset_ids: tuple[str, ...], clip_z: float, min_observations: int) -> Standardizer:
        x = np.asarray(returns, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(asset_ids):
            raise ValueError("returns must be (rows, assets) in canonical asset order")
        if not np.all(np.isfinite(x)):
            raise ValueError("standardizer input must be finite (complete-case rows only)")
        if x.shape[0] < max(min_observations, 2):
            raise InsufficientObservations(f"{x.shape[0]} rows < minimum {min_observations}")
        scale = x.std(axis=0, ddof=1)
        bad = [a for a, s in zip(asset_ids, scale, strict=True) if s < SCALE_FLOOR]
        if bad:
            raise ValueError(f"degenerate scale for {bad}")
        return cls(asset_ids, x.mean(axis=0), scale, clip_z, x.shape[0])

    def transform(self, returns: NDArray[np.float64]) -> NDArray[np.float64]:
        """Unclipped z-scores. The distribution is stored before any clipping (spec 7.2)."""
        return (np.asarray(returns, dtype=float) - self.mean) / self.scale

    def clipped(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(z, -self.clip_z, self.clip_z)


@dataclass(frozen=True)
class PcaFit:
    asset_ids: tuple[str, ...]
    n_components: int
    loadings: NDArray[np.float64]  # (components, assets); rows orthonormal; sign rule applied
    explained_variance_ratio: NDArray[np.float64]
    scores: NDArray[np.float64]  # (rows, components)
    residuals: NDArray[np.float64]  # (rows, assets), standardized units
    standardizer: Standardizer


def apply_sign_rule(components: NDArray[np.float64]) -> NDArray[np.float64]:
    """The loading with the largest absolute value is positive; exact ties resolve to canonical asset order."""
    out = components.copy()
    for k in range(out.shape[0]):
        pivot = int(np.argmax(np.abs(out[k])))  # argmax returns the first maximum = canonical order on ties
        if out[k, pivot] < 0:
            out[k] = -out[k]
    return out


def fit_pca(returns: NDArray[np.float64], asset_ids: tuple[str, ...], n_components: int, clip_z: float,
            min_observations: int) -> PcaFit:
    std = Standardizer.fit(returns, asset_ids, clip_z, min_observations)
    z = std.transform(returns)
    if not 1 <= n_components <= len(asset_ids):
        raise ValueError("n_components out of range")
    model = PCA(n_components=n_components, svd_solver="full").fit(z)
    loadings = apply_sign_rule(model.components_)
    scores = z @ loadings.T
    residuals = z - scores @ loadings
    return PcaFit(asset_ids, n_components, loadings, model.explained_variance_ratio_, scores, residuals, std)


def rolling_first_pc_share(window_returns: NDArray[np.float64], min_observations: int) -> float | None:
    """``largest_eigenvalue / sum(eigenvalues)`` of the window's own standardized covariance (spec 8.3).

    Uses only the supplied trailing window; its own mean/scale are fit inside the window. Returns None when the window
    is too short or degenerate so the caller can report a reason rather than coerce to zero.
    """
    x = np.asarray(window_returns, dtype=float)
    if x.shape[0] < max(min_observations, 2) or not np.all(np.isfinite(x)):
        return None
    scale = x.std(axis=0, ddof=1)
    if np.any(scale < SCALE_FLOOR):
        return None
    z = (x - x.mean(axis=0)) / scale
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False, ddof=1))
    return float(eig[-1] / eig.sum())
