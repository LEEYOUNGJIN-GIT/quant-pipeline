"""Spec section 16 goldens. Evidence class: kernel_unit (explicit inputs; does not prove the fixture path).
Tolerance class: golden_arithmetic (atol=1e-12, rtol=1e-10) unless stated."""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from quant_pipeline.accounting import CostRates, Fill, Ledger
from quant_pipeline.aggregation import score_to_target
from quant_pipeline.fx import krw_return, krw_value
from quant_pipeline.regimes import filter_step, filter_window, predict

GOLDEN = {"abs": 1e-12, "rel": 1e-10}  # for pytest.approx
ALLCLOSE = {"atol": 1e-12, "rtol": 1e-10}  # for numpy


def test_16_1_fx_interaction():
    assert krw_return(0.10, 0.05) == pytest.approx(0.155, **GOLDEN)
    assert krw_return(0.10, 0.05) != pytest.approx(0.15, abs=1e-6)


def test_16_2_score_to_target():
    t = score_to_target(np.array([2.0, -1.0, 1.0]), 0.75, 0.80)
    np.testing.assert_allclose(t, [0.40, 0.00, 0.20, 0.40], **ALLCLOSE)
    neg = score_to_target(np.array([-2.0, 0.0, -1.0]), 0.75, 0.80)
    assert neg.tolist() == [0.0, 0.0, 0.0, 1.0]  # exact all-cash


def test_16_6_filter_step():
    pred, filt = filter_step(np.array([0.6, 0.4]), np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0.3, 0.8]))
    np.testing.assert_allclose(pred, [0.62, 0.38], **ALLCLOSE)
    np.testing.assert_allclose(filt, [0.3795918367346939, 0.6204081632653061], **ALLCLOSE)


def test_16_6_refit_boundary():
    trans = np.array([[0.8, 0.2], [0.1, 0.9]])
    post = filter_window(np.array([0.5, 0.5]), trans, np.array([[0.6, 0.4], [0.3, 0.7], [0.9, 0.2]]))
    np.testing.assert_allclose(post[0], [0.6, 0.4], **ALLCLOSE)
    np.testing.assert_allclose(post[1], [13 / 41, 28 / 41], **ALLCLOSE)
    np.testing.assert_allclose(post[2], [297 / 436, 139 / 436], **ALLCLOSE)
    oos_prior = predict(post[2], trans)
    np.testing.assert_allclose(oos_prior, [503 / 872, 369 / 872], **ALLCLOSE)
    # the stationary distribution [1/3, 2/3] must never be used at the boundary
    stationary = np.array([1 / 3, 2 / 3])
    assert not np.allclose(predict(stationary, trans), oos_prior, atol=1e-6)


def test_16_7_ledoit_wolf_reference():
    x = np.array([
        [0.010, -0.005, 0.002], [0.004, 0.006, -0.001], [-0.003, 0.002, 0.005],
        [0.008, -0.004, 0.003], [-0.002, 0.007, -0.006], [0.005, 0.001, 0.004],
    ])
    cov = np.array([
        [2.103373637856396e-05, -8.486109037833173e-06, 2.694913005257832e-06],
        [-8.486109037833173e-06, 1.978662238144997e-05, -5.518837803320561e-06],
        [2.694913005257832e-06, -5.518837803320561e-06, 1.634630790665274e-05],
    ])
    m = LedoitWolf(assume_centered=False).fit(x)
    # matrix_reference class: atol=1e-12, rtol=1e-8
    np.testing.assert_allclose(m.location_, [0.0036666666667, 0.0011666666667, 0.0011666666667], atol=1e-12, rtol=1e-8)
    assert m.shrinkage_ == pytest.approx(0.4839528287804152, abs=1e-12, rel=1e-8)
    np.testing.assert_allclose(m.covariance_, cov, atol=1e-12, rtol=1e-8)


def test_16_5_post_cost_self_financing():
    ten_bps = CostRates(Decimal(5), Decimal(5), Decimal(0), Decimal(0))
    led = Ledger(cash=Decimal(200000), holdings={"KRX_A": 30, "KRX_B": 20, "NYSE_C": 300})
    price = {"KRX_A": Decimal(10000), "KRX_B": Decimal(10000), "NYSE_C": Decimal(1000)}
    # +10 KRX_A and +5 KRX_B = KRW 150,000 notional; cost 10 bps = KRW 150 (spec: "10 bps on KRW 150,000")
    r1 = led.post_fill(Fill("KRX_A", "KRW", 10, price["KRX_A"], None), ten_bps)
    r2 = led.post_fill(Fill("KRX_B", "KRW", 5, price["KRX_B"], None), ten_bps)
    assert r1["notional"] + r2["notional"] == Decimal("150000.00")
    assert led.cumulative_costs == Decimal("150.00")
    assert led.cash == Decimal("49850.00")
    marks = {a: p for a, p in price.items()}
    nav = led.nav(marks)
    assert nav == Decimal("999850.00")
    w = [Decimal(led.holdings[a]) * marks[a] / nav for a in ("KRX_A", "KRX_B", "NYSE_C")] + [led.cash / nav]
    expected = [0.4000600090, 0.2500375056, 0.3000450068, 0.0498574786]
    np.testing.assert_allclose([float(x) for x in w], expected, atol=1e-10)
    # realized cash weight is below the 5% target floor and that is NOT an error
    assert w[3] < Decimal("0.05") and led.cash >= 0
    # self-financing: NAV falls by exactly the realized costs
    assert Decimal(1000000) - nav == led.cumulative_costs


def test_ledger_rejects_overdraft_and_short_atomically():
    rates = CostRates(Decimal(5), Decimal(5), Decimal(0), Decimal(0))
    led = Ledger(cash=Decimal(1000), holdings={"A": 1})
    with pytest.raises(ValueError):
        led.post_fill(Fill("A", "KRW", 1, Decimal(5000), None), rates)
    with pytest.raises(ValueError):
        led.post_fill(Fill("A", "KRW", -2, Decimal(100), None), rates)
    assert led.cash == Decimal(1000) and led.holdings == {"A": 1} and led.cumulative_costs == 0


def test_usd_fill_uses_krw_per_usd_and_fx_cost():
    rates = CostRates(Decimal(5), Decimal(5), Decimal(0), Decimal(2))
    led = Ledger(cash=Decimal(10000000))
    r = led.post_fill(Fill("NYSE_C", "USD", 10, Decimal(100), Decimal(1300)), rates)
    assert r["notional"] == Decimal("1300000.00")
    assert r["cost"] == Decimal(1300000) * Decimal(12) / Decimal(10000)  # 5+5+2 bps
    assert krw_value(Decimal(10), Decimal(100), "USD", Decimal(1300)) == Decimal(1300000)
    with pytest.raises(ValueError):
        krw_value(Decimal(1), Decimal(1), "USD", None)
    with pytest.raises(ValueError):
        krw_value(Decimal(1), Decimal(1), "KRW", Decimal(1300))
