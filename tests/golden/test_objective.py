"""Objective math and calibration (spec 9.1-9.4, 16.4). Evidence class: kernel_unit with explicit inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_pipeline.config import load_config
from quant_pipeline.objective import (
    calibrate_objective,
    drift,
    expected_cost_rates,
    objective_terms,
    one_way_turnover,
    robust_scale,
    w_track_diagonal,
)

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")

# spec 16.4: two risky assets plus cash
A = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.45, 0.35, 0.2]])
SIGMA = np.diag([0.04, 0.01])


def virtual_path():
    v0 = np.array([0.0, 0.0, 1.0])
    v1 = drift(A[0], np.array([0.10, -0.05]))
    v2 = drift(A[1], np.array([-0.02, 0.04]))
    return np.array([v0, v1, v2])


class TwoAssetCfg:
    """Minimal config view for 16.4: two risky assets at 10 bps each and the profile's calibration constants."""

    def __init__(self):
        self.objective = CFG.objective
        self.costs = CFG.costs.model_copy(update={"commission_bps": 5.0, "half_spread_slippage_bps": 5.0, "fx_conversion_bps": 0.0})
        self.universe = type("U", (), {"assets": [a.model_copy(update={"currency": "KRW", "exchange": "XKRX"})
                                                  for a in (CFG.universe.assets or [])[:2]]})()


def test_virtual_pretrade_path_matches_spec_16_4():
    v = virtual_path()
    np.testing.assert_allclose(v[1], [0.5314009661835749, 0.2753623188405797, 0.1932367149758454], atol=1e-12)
    np.testing.assert_allclose(v[2], [0.3888888888888889, 0.4126984126984127, 0.1984126984126984], atol=1e-12)
    np.testing.assert_allclose(v.sum(axis=1), 1.0, atol=1e-15)  # the full vector is renormalized after drift


def test_w_track_diagonal_and_scales_match_spec_16_4():
    cal = calibrate_objective(TwoAssetCfg(), A, virtual_path(), [SIGMA, SIGMA, SIGMA])
    np.testing.assert_allclose(cal.w_track, [0.7816381611517819, 0.8687686694707877, 1.3495931693774306], atol=1e-12, rtol=1e-10)
    assert cal.w_track.sum() == pytest.approx(3.0, abs=1e-12)  # trace is exactly N + 1
    assert cal.s_track.value == pytest.approx(0.0270535965111216, abs=1e-12, rel=1e-10)
    assert cal.s_risk.value == pytest.approx(0.009325, abs=1e-12, rel=1e-10)
    assert cal.s_cost.value == pytest.approx(0.0002560386473430, abs=1e-12, rel=1e-10)
    assert not (cal.s_track.fallback or cal.s_risk.fallback or cal.s_cost.fallback) and cal.n_decisions == 3


def test_sorted_sample_is_persisted_and_the_median_is_type7():
    s = robust_scale("S", np.array([5.0, 1.0, 3.0, 0.0, np.nan, -2.0]), 1e-9)
    assert s.sorted_sample == (1.0, 3.0, 5.0) and s.value == 3.0 and s.n_positive == 3 and not s.fallback
    even = robust_scale("S", np.array([4.0, 1.0, 2.0, 3.0]), 1e-9)
    assert even.value == 2.5  # Type-7 interpolation between the two middle values


def test_no_positive_value_uses_the_floor_and_sets_the_fallback_flag():
    s = robust_scale("S_risk", np.array([0.0, 0.0, np.nan]), 1e-10)
    assert s.value == 1e-10 and s.fallback and s.sorted_sample == ()


def test_w_track_clips_extreme_weights_and_keeps_the_trace():
    deltas = np.array([[1e-9, 0.5, 0.0], [1e-9, 0.4, 0.0]])  # tiny, large and exactly-zero deltas
    w = w_track_diagonal(deltas, floor=1e-6, w_min=0.01, w_max=100.0)
    assert w.sum() == pytest.approx(3.0) and w[0] == pytest.approx(w[2])  # both hit the 1/floor clip, then the max clip
    assert w[1] < w[0]


def test_zero_observations_are_included_in_the_median():
    d = np.array([[0.0], [0.0], [0.3]])
    assert w_track_diagonal(d, 1e-6, 0.01, 100.0)[0] == pytest.approx(1.0)  # median(delta^2) = 0 -> floor -> clipped, trace N+1 = 1


def test_expected_cost_rates_follow_the_profile_and_cash_is_free():
    r = expected_cost_rates(CFG)
    assert r[-1] == 0.0 and r[0] == r[1] == pytest.approx(10e-4) and r[2] == pytest.approx(12e-4)  # USD adds 2 bp FX
    np.testing.assert_allclose(expected_cost_rates(CFG, 2.0), 2 * r)


def test_one_way_turnover_is_half_the_full_vector_l1_including_cash():
    a, b = np.array([0.2, 0.3, 0.5]), np.array([0.0, 0.0, 1.0])
    assert one_way_turnover(a, b) == pytest.approx(0.5)  # from 100 % cash, turnover equals the risky weight bought


def test_objective_terms_reconcile_with_a_hand_calculation():
    w, a, wm = np.array([0.3, 0.2, 0.5]), np.array([0.4, 0.1, 0.5]), np.array([0.2, 0.2, 0.6])
    W, rates = np.array([1.0, 2.0, 3.0]), np.array([0.001, 0.002, 0.0])
    t = objective_terms(w, a, wm, W, np.diag([0.04, 0.01]), rates, (0.5, 0.25, 0.01), (1.0, 2.0, 3.0))
    track = 1.0 * 0.01 + 2.0 * 0.01 + 0.0
    risk = 0.3 ** 2 * 0.04 + 0.2 ** 2 * 0.01
    cost = 0.001 * 0.1 + 0.0 + 0.0
    assert t.track == pytest.approx(track) and t.risk == pytest.approx(risk) and t.cost == pytest.approx(cost)
    assert t.contributions == pytest.approx((track / 0.5, 2 * risk / 0.25, 3 * cost / 0.01))
    assert t.objective == pytest.approx(sum(t.contributions))


def test_s_cost_is_always_the_one_x_calibration():
    cal = calibrate_objective(TwoAssetCfg(), A, virtual_path(), [SIGMA] * 3)
    rates1 = expected_cost_rates(CFG)
    assert rates1.shape == (4,) and cal.s_cost.value == pytest.approx(0.0002560386473430, rel=1e-9)  # not rescaled by a multiplier


def test_a_row_without_covariance_history_is_excluded_with_a_reason_not_backfilled():
    cal = calibrate_objective(TwoAssetCfg(), A, virtual_path(), [None, SIGMA, SIGMA])
    assert cal.excluded == ((0, "insufficient_covariance_history"),) and cal.s_risk.n_positive == 2
