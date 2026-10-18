"""Agents, score transformation, gates/mixture/envelope (spec 8.1-8.3, 16.2, 16.3) and risk-pressure thresholds.
Evidence class: kernel_unit with explicit inputs."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from quant_pipeline.agents import (
    AGENTS,
    CASH,
    AgentContext,
    AgentOutput,
    defensive,
    diversification,
    mean_reversion,
    trend,
)
from quant_pipeline.agents.base import transform_scores
from quant_pipeline.aggregation import combine, defensive_envelope, normalize_gates, raw_gates
from quant_pipeline.config import load_config
from quant_pipeline.risk_pressure import calibrate, metric_series, pressure, type7_quantile

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
ASSETS = ("KRX_A", "KRX_B", "NYSE_C")
NOW = datetime(2024, 3, 1, tzinfo=UTC)
GATES = {"low_risk": {"Trend": 1.20, "MeanReversion": 1.00, "Defensive": 0.60, "Diversification": 1.00},
         "high_risk": {"Trend": 0.50, "MeanReversion": 0.70, "Defensive": 1.50, "Diversification": 1.20}}


def ctx(returns, tradable=(True, True, True), max_weights=(0.6, 0.6, 0.6), cov=None, loadings=None, rows=None):
    r = np.asarray(returns, dtype=float)
    cov = np.diag([0.0004, 0.0001, 0.0002]) if cov is None else np.asarray(cov, dtype=float)
    load = np.array([[0.58, 0.58, 0.58]]) if loadings is None else np.asarray(loadings, dtype=float)
    return AgentContext("f", "r", NOW, ASSETS, r, np.array(tradable), load, r.mean(axis=0), r.std(axis=0, ddof=1) + 1e-9,
                        cov, np.array(max_weights, dtype=float), len(r) - 1, 12 if rows is None else rows)


def noise(n=60, seed=1, scale=(0.01, 0.005, 0.008)):
    return np.random.default_rng(seed).normal(0, 1, (n, 3)) * np.array(scale)


def output(name, shares, budget):
    t = {a: budget * s for a, s in zip(ASSETS, shares, strict=True)}
    t[CASH] = 1 - budget
    return AgentOutput("f", "r", NOW, name, dict(zip(ASSETS, shares, strict=True)), dict(zip(ASSETS, shares, strict=True)),
                       1.0, budget, t, 0.0, {"agent_status": "ok"})


# ----------------------------------------------------------------------------------- 8.1 / 16.2
def test_score_transform_matches_spec_16_2_and_never_shorts():
    norm, budget, _ = transform_scores(np.array([2.0, -1.0, 1.0]), np.ones(3, bool), 0.75, 0.80, 1e-12)
    np.testing.assert_allclose(norm, [2 / 3, 0, 1 / 3])
    assert budget == pytest.approx(0.60)  # was silently skipped by an `and` chain
    n2, b2, _ = transform_scores(np.array([-2.0, 0.0, -1.0]), np.ones(3, bool), 0.75, 0.80, 1e-12)
    assert not n2.any() and b2 == 0.0  # all-cash, no abs() and no softmax of negatives
    n3, _, _ = transform_scores(np.array([5.0, 1.0, 1.0]), np.array([False, True, True]), 1.0, 1.0, 1e-12)
    assert n3[0] == 0.0 and n3.sum() == pytest.approx(1.0)  # an ineligible asset gets nothing even with the top score
    assert transform_scores(np.array([1.0, 0, 0]), np.ones(3, bool), 7.0, 0.8, 1e-12)[2] == 1.0  # confidence clipped
    n4, b4, _ = transform_scores(np.array([1e-13, 0, 0]), np.ones(3, bool), 1.0, 0.8, 1e-12)
    assert not n4.any() and b4 == 0.0  # mass at or below the floor is treated as no signal
    with pytest.raises(ValueError):
        transform_scores(np.array([np.nan, 0, 0]), np.ones(3, bool), 1.0, 0.8, 1e-12)


@pytest.mark.parametrize("run", [trend.run, mean_reversion.run, defensive.run, diversification.run])
def test_every_agent_returns_a_valid_cash_inclusive_target(run):
    out = run(ctx(noise()), CFG)
    v = out.vector(ASSETS)
    assert v.shape == (4,) and (v >= 0).all() and v.sum() == pytest.approx(1.0, abs=1e-12)
    assert out.proposed_risky_budget == pytest.approx(v[:3].sum(), abs=1e-12) and 0 <= out.confidence <= 1
    assert out.normalized_score.keys() == out.raw_score.keys() == set(ASSETS)


# ----------------------------------------------------------------------------------- Trend
def test_trend_uses_only_rows_up_to_the_decision_and_prefers_the_uptrend():
    r = noise(80, scale=(0.01, 0.01, 0.01))
    r[:, 0] += 0.02  # a persistent, unmistakable drift in asset 0 only (trend uses the last 2/5/10 rows)
    f = CFG.features
    full, _ = trend.scores_at(r, 59, f.return_horizons, [0.2, 0.3, 0.5], f.volatility_window, f.clip_z, f.min_observations)
    prefix, _ = trend.scores_at(r[:60], 59, f.return_horizons, [0.2, 0.3, 0.5], f.volatility_window, f.clip_z, f.min_observations)
    np.testing.assert_array_equal(full, prefix)  # rows after t are invisible
    assert full.argmax() == 0 and full[0] > 0 and np.abs(full).max() <= f.clip_z


def test_trend_reports_zero_coverage_and_goes_to_cash_without_history():
    out = trend.run(ctx(noise(5)), CFG)
    assert out.diagnostics["horizon_coverage"] == 0.0 and out.proposed_risky_budget == 0.0
    assert out.absolute_target[CASH] == 1.0


def test_trend_confidence_scales_with_positive_mass_against_a_training_only_reference():
    out = trend.run(ctx(noise(70, seed=4)), CFG)
    assert out.diagnostics["reference_positive_mass"] >= CFG.features.score_mass_floor
    assert out.confidence == pytest.approx(out.diagnostics["horizon_coverage"] * min(1.0, out.diagnostics["positive_mass"]
                                                                              / out.diagnostics["reference_positive_mass"]))
    assert out.diagnostics["calibration_rows"] == [12, 69]  # training rows only; nothing past train_end_row feeds the reference


# ----------------------------------------------------------------------------------- MeanReversion
def test_mean_reversion_state_machine_has_entry_exit_hysteresis():
    z = np.array([[0.0], [-1.2], [-0.6], [-0.1], [-0.5]])
    seq = [bool(mean_reversion.replay_state(z[: i + 1], 1.0, 0.25)[0]) for i in range(len(z))]
    assert seq == [False, True, True, False, False]  # opens at <= -1, stays until >= -0.25, -0.5 does not re-enter


def test_mean_reversion_goes_long_only_the_underperforming_asset():
    r = noise(60, seed=9, scale=(0.004, 0.004, 0.004))
    r[-1, 1] = -0.05  # asset 1 drops sharply on the decision row
    out = mean_reversion.run(ctx(r, loadings=np.zeros((1, 3))), CFG, entry_z=2.5)  # a lone outlier in a 10-row window reaches at most z = 9/sqrt(10) ~ 2.85
    assert out.diagnostics["active"] == [False, True, False]
    assert out.raw_score["KRX_B"] > 0 and out.raw_score["KRX_A"] == out.raw_score["NYSE_C"] == 0.0
    assert out.diagnostics["active_entry_fraction"] == pytest.approx(1 / 3)
    assert out.proposed_risky_budget == pytest.approx(CFG.agents.mean_reversion.risk_budget_cap / 3 * out.diagnostics["data_coverage"])


# ----------------------------------------------------------------------------------- Defensive
def test_defensive_prefers_lower_downside_and_less_drawdown_and_excludes_untradable():
    r = noise(60, seed=2, scale=(0.004, 0.004, 0.004))
    r[:, 0] = np.where(r[:, 0] < 0, r[:, 0] * 4, r[:, 0])  # heavy downside in asset 0
    out = defensive.run(ctx(r), CFG)
    assert out.raw_score["KRX_A"] < out.raw_score["KRX_B"] and out.raw_score["KRX_A"] < out.raw_score["NYSE_C"]
    cut = defensive.run(ctx(r, tradable=(True, False, True)), CFG)
    assert cut.normalized_score["KRX_B"] == 0.0 and cut.confidence == pytest.approx(2 / 3)
    assert out.proposed_risky_budget <= CFG.agents.defensive.risk_budget_cap


def test_defensive_penalizes_common_factor_loading():
    r = noise(60, seed=6, scale=(0.005, 0.005, 0.005))
    flat = defensive.run(ctx(r, loadings=[[0.58, 0.58, 0.58]]), CFG)
    tilted = defensive.run(ctx(r, loadings=[[0.9, 0.1, 0.1]]), CFG)
    assert tilted.raw_score["KRX_A"] < flat.raw_score["KRX_A"] and tilted.raw_score["KRX_B"] > flat.raw_score["KRX_B"]


# ----------------------------------------------------------------------------------- Diversification
def test_diversification_matches_the_closed_form_when_no_cap_binds():
    out = diversification.run(ctx(noise(), cov=np.diag([0.04, 0.01, 0.02]), max_weights=(1.0, 1.0, 1.0)), CFG)
    inv = 1 / np.array([0.04, 0.01, 0.02])
    np.testing.assert_allclose(list(out.normalized_score.values()), inv / inv.sum(), atol=1e-4)
    assert out.diagnostics["agent_status"] == "ok" and out.diagnostics["solver"]["attempts"][0]["status"] == "optimal"


def test_diversification_cap_active_case_pins_the_low_variance_asset_at_its_sleeve_cap():
    out = diversification.run(ctx(noise(), cov=np.diag([0.04, 0.0001, 0.02])), CFG)
    cap = 0.6 / 0.8  # sleeve cap = max_weight / B_div_effective
    assert out.normalized_score["KRX_B"] == pytest.approx(cap, abs=1e-6)
    assert out.absolute_target["KRX_B"] <= 0.6 + 1e-9  # the cap is rechecked in portfolio-weight units
    assert sum(out.normalized_score.values()) == pytest.approx(1.0)


def test_runtime_clipping_reduces_the_budget_to_the_eligible_cap_sum_and_is_reported():
    out = diversification.run(ctx(noise(), max_weights=(0.2, 0.2, 0.2)), CFG)
    d = out.diagnostics
    assert d["B_div_proposed"] == pytest.approx(0.8) and d["eligible_cap_sum"] == pytest.approx(0.6)
    assert d["B_div_effective"] == pytest.approx(0.6)
    assert d["B_div_effective"] < d["B_div_proposed"]  # clipped below the proposal
    assert out.absolute_target[CASH] == pytest.approx(0.4) and max(out.absolute_target[a] for a in ASSETS) <= 0.2 + 1e-9


def test_a_tiny_budget_degrades_the_agent_instead_of_silently_going_to_cash():
    out = diversification.run(ctx(noise(), max_weights=(0.01, 0.01, 0.01)), CFG)
    assert out.diagnostics["agent_status"] == "degraded_no_meaningful_budget" and out.confidence == 0.0
    assert out.diagnostics["reason"] and out.absolute_target[CASH] == 1.0
    none = diversification.run(ctx(noise(), tradable=(False, False, False)), CFG)
    assert none.diagnostics["agent_status"] == "degraded_no_meaningful_budget"


def test_solver_failure_uses_the_declared_inverse_volatility_fallback(monkeypatch):
    monkeypatch.setattr(diversification, "solve_minvar", lambda s, c, cfg: (None, {"attempts": []}))
    out = diversification.run(ctx(noise(), cov=np.diag([0.04, 0.01, 0.02])), CFG)
    assert out.diagnostics["method"] == "inverse_volatility_capped_simplex" and out.diagnostics["fallback_used"]
    assert sum(out.normalized_score.values()) == pytest.approx(1.0)


def test_capped_simplex_properties():
    q = diversification.capped_simplex(np.array([5.0, 1.0, 1.0]), np.array([0.5, 1.0, 1.0]))
    assert q.sum() == pytest.approx(1.0) and q[0] == pytest.approx(0.5) and q[1] == pytest.approx(0.25)
    assert diversification.capped_simplex(np.ones(3), np.full(3, 0.3)) is None  # infeasible: caps sum below one
    np.testing.assert_allclose(diversification.capped_simplex(np.ones(4), np.ones(4)), 0.25)


def test_defensive_and_diversification_are_not_aliases():
    assert defensive.run is not diversification.run and defensive.run.__code__ is not diversification.run.__code__
    c = ctx(noise())
    d1, d2 = defensive.run(c, CFG), diversification.run(c, CFG)
    assert set(d1.diagnostics) != set(d2.diagnostics)  # different method, different persisted diagnostics
    assert "downside" in d1.diagnostics and "downside" not in d2.diagnostics and "ridge" in d2.diagnostics


# ----------------------------------------------------------------------------------- 8.3 gates / mixture / envelope
def _kw(**over):
    base = {"p_high": 0.75, "pressure_components": {"drawdown": 0.1, "factor": 0.2, "liquidity": 0.0}, "base": 0.8,
            "minimum": 0.2, "gamma": 1.0, "target_cash_floor": 0.05, "max_weights": np.full(3, 0.6),
            "usd_mask": np.array([False, False, True]), "usd_limit": 0.5}
    return base | over


GOLDEN_SHARES = {"Trend": [2 / 3, 0, 1 / 3], "MeanReversion": [0, 1, 0], "Defensive": [0.2, 0.3, 0.5],
                 "Diversification": [0.3, 0.3, 0.4]}
GOLDEN_BUDGETS = {"Trend": 0.6, "MeanReversion": 0.5, "Defensive": 0.4, "Diversification": 0.8}


def golden_outputs():
    return [output(k, GOLDEN_SHARES[k], GOLDEN_BUDGETS[k]) for k in AGENTS]


def test_spec_16_3_gates_budget_and_defensive_envelope():
    g = raw_gates(np.array([0.25, 0.75]), ["low_risk", "high_risk"], GATES)
    np.testing.assert_allclose(g, [0.675, 0.775, 1.275, 1.150], atol=1e-12)
    np.testing.assert_allclose(normalize_gates(g, np.zeros(4, bool)),
                               [0.1741935483870968, 0.2, 0.3290322580645161, 0.2967741935483871], atol=1e-15)
    c = combine(golden_outputs(), ASSETS, np.array([0.25, 0.75]), ["low_risk", "high_risk"], GATES, **_kw())
    assert c.b_proposed == pytest.approx(0.5735483870967741, abs=1e-15)
    assert c.b_defensive == pytest.approx(0.35, abs=1e-15) and c.b_final == pytest.approx(0.35, abs=1e-15)
    np.testing.assert_allclose([c.a_bar[a] for a in ASSETS], [0.0948387096774194, 0.1357096774193548, 0.1194516129032258],
                               atol=1e-15)
    assert c.a_bar[CASH] == pytest.approx(0.65) and sum(c.a_bar.values()) == pytest.approx(1.0)
    assert sum(c.a_bar[a] for a in ASSETS) <= 0.35 + 1e-15  # no later operation raises gross above the envelope


def test_there_is_no_risky_only_renormalization():
    c = combine(golden_outputs(), ASSETS, np.array([0.25, 0.75]), ["low_risk", "high_risk"], GATES, **_kw())
    gross = sum(c.a_bar[a] for a in ASSETS)
    assert gross == pytest.approx(c.b_final) and gross < 0.5
    assert c.a_bar[CASH] == pytest.approx(1 - c.b_final)
    assert gross != pytest.approx(1.0)  # the risky sleeve is not rescaled up to one


def test_defensive_overlay_raises_cash_as_pressure_rises():
    low = combine(golden_outputs(), ASSETS, np.array([0.9, 0.1]), ["low_risk", "high_risk"], GATES,
                  **_kw(p_high=0.1, pressure_components={"d": 0.0}))
    high = combine(golden_outputs(), ASSETS, np.array([0.1, 0.9]), ["low_risk", "high_risk"], GATES,
                   **_kw(p_high=0.9, pressure_components={"d": 0.0}))
    assert high.a_bar[CASH] > low.a_bar[CASH] and high.b_defensive < low.b_defensive
    assert defensive_envelope(0.0, 0.8, 0.2, 1.0) == 0.8 and defensive_envelope(1.0, 0.8, 0.2, 1.0) == 0.2


def test_any_single_pressure_component_can_drive_the_envelope():
    c = combine(golden_outputs(), ASSETS, np.array([0.9, 0.1]), ["low_risk", "high_risk"], GATES,
                **_kw(p_high=0.1, pressure_components={"factor": 0.95}))
    assert c.risk_pressure == 0.95 and c.b_final == pytest.approx(0.2 + 0.6 * 0.05)


def test_a_degraded_agent_abstains_and_gates_are_renormalized():
    outs = golden_outputs()
    outs[3] = AgentOutput("f", "r", NOW, "Diversification", dict.fromkeys(ASSETS, 0.0), dict.fromkeys(ASSETS, 0.0), 0.0, 0.0,
                          {**dict.fromkeys(ASSETS, 0.0), CASH: 1.0}, 0.0, {"agent_status": "degraded_no_meaningful_budget"})
    c = combine(outs, ASSETS, np.array([0.25, 0.75]), ["low_risk", "high_risk"], GATES, **_kw())
    assert c.abstaining == ("Diversification",) and c.gates_normalized[3] == 0.0 and sum(c.gates_normalized) == pytest.approx(1.0)
    assert c.b_final > 0  # one agent's infeasibility does not switch the portfolio to cash


def test_all_agents_abstaining_gives_zero_risk():
    dead = [AgentOutput("f", "r", NOW, k, dict.fromkeys(ASSETS, 0.0), dict.fromkeys(ASSETS, 0.0), 0.0, 0.0,
                        {**dict.fromkeys(ASSETS, 0.0), CASH: 1.0}, 0.0, {"agent_status": "degraded_no_meaningful_budget"})
            for k in AGENTS]
    c = combine(dead, ASSETS, np.array([0.5, 0.5]), ["low_risk", "high_risk"], GATES, **_kw())
    assert c.b_final == 0.0 and c.a_bar[CASH] == 1.0 and c.gates_normalized is None


def test_all_zero_gates_use_the_named_fallback_and_unknown_fallback_fails():
    np.testing.assert_allclose(normalize_gates(np.zeros(4), np.array([False, False, True, False])), [1 / 3, 1 / 3, 0, 1 / 3])
    with pytest.raises(ValueError):
        normalize_gates(np.zeros(4), np.zeros(4, bool), fallback="softmax")


def test_caps_only_reduce_risk_and_leave_the_residual_in_cash():
    c = combine(golden_outputs(), ASSETS, np.array([0.25, 0.75]), ["low_risk", "high_risk"], GATES,
                **_kw(max_weights=np.array([0.05, 0.05, 0.05])))
    assert max(c.a_bar[a] for a in ASSETS) <= 0.05 and c.a_bar[CASH] == pytest.approx(1 - sum(c.a_bar[a] for a in ASSETS))
    assert c.diagnostics["cap_reduction"] > 0
    usd = combine(golden_outputs(), ASSETS, np.array([0.25, 0.75]), ["low_risk", "high_risk"], GATES, **_kw(usd_limit=0.05))
    assert usd.a_bar["NYSE_C"] == pytest.approx(0.05)


def test_gate_table_label_mismatch_is_an_error():
    with pytest.raises(ValueError):
        raw_gates(np.array([0.5, 0.5]), ["low_risk"], GATES)
    with pytest.raises(KeyError):
        raw_gates(np.array([0.5, 0.5]), ["low_risk", "medium_risk"], GATES)


# ----------------------------------------------------------------------------------- risk-pressure thresholds
def test_type7_quantile_matches_hand_computation_and_pressure_is_clipped():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert type7_quantile(x, 0.5) == 3.0 and type7_quantile(x, 0.9) == pytest.approx(4 + 0.6 * 6)
    assert pressure(5, 2, 4, 1e-6) == 1.0 and pressure(1, 2, 4, 1e-6) == 0.0 and pressure(3, 2, 4, 1e-6) == 0.5


def test_thresholds_are_fitted_on_training_values_only_or_fall_back_with_a_reason():
    kw = {"start_q": 0.5, "full_q": 0.9, "gap_floor": 1e-6, "min_decisions": 8, "fallback": (0.05, 0.2)}
    t = calibrate("m", np.linspace(0, 1, 20), **kw)
    assert not t.fallback and t.start == pytest.approx(0.5) and t.full == pytest.approx(0.9) and t.n_training == 20
    short = calibrate("m", np.arange(5.0), **kw)
    assert short.fallback and (short.start, short.full) == (0.05, 0.2) and short.reason == "insufficient_training_decisions"
    flat = calibrate("m", np.ones(20), **kw)
    assert flat.fallback and flat.reason == "insufficient_spread"
    assert calibrate("m", np.r_[np.linspace(0, 1, 20), np.nan], **kw).n_training == 20  # non-finite values are ignored


def test_metric_series_value_at_a_row_never_depends_on_later_rows():
    r = noise(70, seed=8)
    ill = np.zeros(70)
    kw = {"drawdown_window": 20, "factor_window": 20, "factor_min_obs": 12, "first_row": 12}
    full = metric_series(r, ill, **kw)
    head = metric_series(r[:50], ill[:50], **kw)
    for m in full:
        np.testing.assert_array_equal(full[m][:50], head[m])  # a fold-end value is never copied backward
