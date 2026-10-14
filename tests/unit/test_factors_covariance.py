"""Kernel-level checks for standardization, PCA, the rolling pressure statistic and Ledoit-Wolf (kernel_unit evidence)."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from quant_pipeline.covariance import PsdRepairError, fit_ledoit_wolf, repair_psd
from quant_pipeline.factors import (
    InsufficientObservations,
    Standardizer,
    apply_sign_rule,
    fit_pca,
    rolling_first_pc_share,
)

RNG = np.random.default_rng(7)
ASSETS = ("A", "B", "C")
X = RNG.normal(size=(60, 3)) @ np.array([[1.0, 0.6, 0.2], [0.0, 0.8, 0.3], [0.0, 0.0, 0.5]]) * 0.01


def test_standardizer_matches_numpy_and_stores_unclipped_before_clipping():
    s = Standardizer.fit(X, ASSETS, clip_z=1.0, min_observations=8)
    z = s.transform(X)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0, ddof=1), 1, atol=1e-12)
    assert np.abs(z).max() > 1.0 and np.abs(s.clipped(z)).max() == 1.0  # clipping is a separate, later step


@pytest.mark.parametrize("bad,exc", [(X[:5], InsufficientObservations), (np.where(np.arange(60)[:, None] >= 0, 0.0, X), ValueError),
                                     (np.full((60, 3), np.nan), ValueError)])
def test_standardizer_rejects_short_degenerate_or_nonfinite_input(bad, exc):
    with pytest.raises(exc):
        Standardizer.fit(bad, ASSETS, 4.0, 8)


def test_pca_loadings_orthonormal_reconstruct_and_match_eigendecomposition():
    p = fit_pca(X, ASSETS, 2, 4.0, 8)
    np.testing.assert_allclose(p.loadings @ p.loadings.T, np.eye(2), atol=1e-12)
    z = p.standardizer.transform(X)
    np.testing.assert_allclose(p.scores @ p.loadings + p.residuals, z, atol=1e-12)
    np.testing.assert_allclose(p.scores.T @ p.residuals, 0, atol=1e-10)  # residuals orthogonal to the factor space
    w, v = np.linalg.eigh(np.cov(z, rowvar=False, ddof=1))  # independent reference
    np.testing.assert_allclose(p.explained_variance_ratio, (w[::-1] / w.sum())[:2], atol=1e-12)
    np.testing.assert_allclose(np.abs(p.loadings[0]), np.abs(v[:, -1]), atol=1e-8)


def test_sign_rule_largest_absolute_loading_positive_and_ties_resolve_to_canonical_order():
    out = apply_sign_rule(np.array([[0.2, -0.9, 0.1], [-0.5, 0.5, 0.0], [0.3, 0.1, -0.2]]))
    assert out[0, 1] > 0 and out[1, 0] > 0 and out[2, 0] > 0  # tie in row 1 -> first asset decides
    for row in out:
        assert row[int(np.argmax(np.abs(row)))] > 0


def test_pca_is_deterministic_and_sign_stable_under_flipping_the_data():
    a, b = fit_pca(X, ASSETS, 1, 4.0, 8), fit_pca(-X, ASSETS, 1, 4.0, 8)
    np.testing.assert_allclose(a.loadings, b.loadings, atol=1e-10)  # the sign rule removes the arbitrary sign


def test_rolling_share_uses_only_the_given_window_and_reports_none_when_undefined():
    share = rolling_first_pc_share(X[-20:], 12)
    assert share is not None and 1 / 3 <= share <= 1.0
    assert rolling_first_pc_share(X[-5:], 12) is None
    assert rolling_first_pc_share(np.zeros((20, 3)), 12) is None  # constant window: undefined, never coerced to 0
    assert rolling_first_pc_share(X[-20:], 12) == share  # no hidden state


GOLDEN_ROWS = np.array([[0.010, -0.005, 0.002], [0.004, 0.006, -0.001], [-0.003, 0.002, 0.005],
                        [0.008, -0.004, 0.003], [-0.002, 0.007, -0.006], [0.005, 0.001, 0.004]])


def _lw(x, **kw):
    t = datetime(2024, 1, 1, tzinfo=UTC)
    args = {"fold_id": "f", "refit_id": "r", "fit_start_utc": t, "fit_end_utc": t, "effective_observations": {},
            "min_observations": 3, "eigen_floor": -1e-12, "max_frobenius": 1e-10} | kw
    return fit_ledoit_wolf(x, ASSETS, **args)


def test_production_wrapper_reproduces_the_spec_16_7_golden():
    r = _lw(GOLDEN_ROWS)
    assert r.diagnostics["shrinkage"] == pytest.approx(0.4839528287804152, abs=1e-12)
    assert r.matrix[0, 0] == pytest.approx(2.103373637856396e-05, rel=1e-8) and r.matrix[1, 2] == pytest.approx(-5.518837803320561e-06, rel=1e-8)
    assert r.estimator_name.startswith("LedoitWolf") and r.n_observations == 6
    assert 0.0 <= r.diagnostics["shrinkage"] <= 1.0 and r.diagnostics["psd"]["repaired"] is False
    assert np.all(np.linalg.eigvalsh(r.matrix) > 0) and r.diagnostics["condition_number"] > 1


def test_psd_repair_is_recorded_and_oversized_repair_fails_the_baseline():
    bad = np.array([[1.0, 0.0], [0.0, -1e-6]])
    fixed, info = repair_psd(bad, -1e-12, 1e-3)
    assert info["repaired"] and info["frobenius_repair"] == pytest.approx(1e-6) and np.linalg.eigvalsh(fixed).min() >= -1e-15
    with pytest.raises(PsdRepairError):
        repair_psd(bad, -1e-12, 1e-9)


def test_covariance_rejects_nonfinite_short_or_misshaped_input():
    with pytest.raises(ValueError):
        _lw(np.where(np.arange(18).reshape(6, 3) == 4, np.nan, GOLDEN_ROWS))
    with pytest.raises(ValueError):
        _lw(GOLDEN_ROWS[:2])
    with pytest.raises(ValueError):
        _lw(GOLDEN_ROWS[:, :2])
