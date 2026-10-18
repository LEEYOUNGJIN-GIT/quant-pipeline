"""Optimal-weights comparison for the weekly / alternate-anchor sensitivities (INV-ASYNC-001, spec 5.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_pipeline.agents.diversification import solve_minvar
from quant_pipeline.asynchrony import _weights_comparison, min_variance_weights
from quant_pipeline.config import load_config

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
SIGMA = np.array([[0.04, 0.006, 0.0], [0.006, 0.01, 0.002], [0.0, 0.002, 0.0225]])


def test_two_independent_solvers_agree_on_the_minimum_variance_weights():
    n = len(SIGMA)
    osqp_w, _ = solve_minvar(SIGMA + 1e-8 * np.eye(n), np.ones(n), CFG)
    assert osqp_w is not None and np.allclose(min_variance_weights(SIGMA), osqp_w, atol=2e-4)


def test_the_closed_form_two_asset_case_is_reproduced():
    s = np.array([[0.04, 0.0], [0.0, 0.01]])  # w1 = var2 / (var1 + var2) = 0.2
    assert min_variance_weights(s) == pytest.approx([0.2, 0.8], abs=1e-5)


def test_weights_are_scale_free_long_only_and_sum_to_one():
    w = min_variance_weights(SIGMA)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0) and np.allclose(w, min_variance_weights(SIGMA * 7.3), atol=1e-6)


def test_a_different_covariance_gives_different_weights_and_the_comparison_reports_the_difference():
    other = SIGMA.copy()
    other[0, 1] = other[1, 0] = 0.0095
    cmp = _weights_comparison(SIGMA, other, ["A", "B", "C"])
    assert cmp["max_abs_weight_diff"] > 1e-3 and cmp["one_way_turnover_between_them"] > 0
    same = _weights_comparison(SIGMA, SIGMA, ["A", "B", "C"])
    assert same["max_abs_weight_diff"] == 0.0


def test_a_failing_solve_is_an_explicit_error_not_a_silent_default(monkeypatch):
    import quant_pipeline.asynchrony as mod

    class Bad:
        success, message, x = False, "no", np.zeros(3)

    monkeypatch.setattr(mod, "minimize", lambda *a, **k: Bad())
    with pytest.raises(ValueError):
        mod.min_variance_weights(SIGMA)
