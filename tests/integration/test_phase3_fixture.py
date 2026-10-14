"""Phase 3 gate (spec 19) on the real fixture: agents, fixed regime table, gates, pressure thresholds and the
Defensive envelope. Evidence class: fixture_path."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.agents import AGENTS, CASH, defensive, diversification, mean_reversion, trend
from quant_pipeline.aggregation import combine
from quant_pipeline.calendars import Calendars, build_row_map
from quant_pipeline.config import load_config
from quant_pipeline.data import load_market_data
from quant_pipeline.fitstate import (
    HMM_FEATURE_ORDER,
    agent_context,
    decision_cutoff,
    fit_core,
    hmm_features,
    lineage_hashes,
)
from quant_pipeline.regimes import FixedRegimeTable, run_hmm_pipeline
from quant_pipeline.risk_pressure import calibrate, metric_series, pressure

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {"Trend": trend.run, "MeanReversion": mean_reversion.run, "Defensive": defensive.run,
           "Diversification": diversification.run}
LOW_POSITIONS, HIGH_POSITIONS = (10, 31, 41, 50, 70), (45, 58, 60)  # per the fixture regime blocks


@pytest.fixture(scope="module")
def env():
    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    cals = Calendars.load(["XKRX", "XNYS"])
    rm, sess = build_row_map(cfg, cals)
    return {"cfg": cfg, "cals": cals, "rm": rm, "sess": sess, "md": load_market_data(cfg, ROOT),
            "hashes": lineage_hashes(cfg, ROOT),
            "table": FixedRegimeTable.load(ROOT / cfg.regime.fixed_table.path), "cache": {}}


def at(env, position, md=None):
    """Fit at the decision's own cutoff (a refit at every decision is a superset of the refit schedule) and the context."""
    cfg, cals = env["cfg"], env["cals"]
    key = (position, id(md))
    if key not in env["cache"]:
        cut = decision_cutoff(cals, cfg, env["rm"].eligible[position - 1])
        fit = fit_core(cfg, md or env["md"], cals, env["sess"], fold_id="f", refit_id=f"r{position}", fit_end_utc=cut,
                       hashes=env["hashes"])
        ctx = agent_context(cfg, md or env["md"], cals, env["sess"], fit, cut)
        env["cache"][key] = (cut, fit, ctx)
    return env["cache"][key]


def run_all(ctx, cfg):
    return [RUNNERS[k](ctx, cfg) for k in AGENTS]


def pressure_components(env, ctx, fit):
    cfg, rp = env["cfg"], env["cfg"].risk_pressure_calibration
    ill = np.zeros(len(ctx.returns))  # no illiquid history in the pressure sample for this synthetic fixture
    series = metric_series(ctx.returns, ill, drawdown_window=cfg.features.drawdown_window,
                           factor_window=rp.factor_window_rows, factor_min_obs=rp.factor_min_observations,
                           first_row=ctx.train_start_row)
    out = {}
    last = len(ctx.returns) - 1
    for name, fb in (("market_drawdown_magnitude", rp.fallback["drawdown"]), ("first_pca_variance_share", rp.fallback["factor"]),
                     ("illiquid_fraction", rp.fallback["liquidity"])):
        train = series[name][ctx.train_start_row: ctx.train_end_row + 1]
        th = calibrate(name, train, start_q=rp.start_quantile, full_q=rp.full_quantile, gap_floor=rp.threshold_gap_floor,
                       min_decisions=rp.minimum_training_decisions, fallback=tuple(fb))
        x = float(ctx.tradable.size - ctx.tradable.sum()) / ctx.tradable.size if name == "illiquid_fraction" else series[name][last]
        out[name] = (pressure(x, th.start, th.full, rp.threshold_gap_floor) if np.isfinite(x) else 0.0, th)
    return out


def test_all_agents_emit_valid_cash_inclusive_targets_on_fixture_decisions(env):
    for pos in (10, 28, 31, 41, 50, 60, 70):
        _, _, ctx = at(env, pos)
        for o in run_all(ctx, env["cfg"]):
            v = o.vector(ctx.assets)
            assert (v >= -1e-12).all() and v.sum() == pytest.approx(1.0, abs=1e-12), (pos, o.agent_name)
            # per-asset caps are applied after the mixture (spec 8.3); an agent is bound only by its own risk-budget cap,
            # except Diversification whose sleeve constraints already imply the asset caps
            assert v[:-1].sum() <= getattr(env["cfg"].agents, o.agent_name.lower().replace("meanreversion", "mean_reversion")
                                           ).risk_budget_cap + 1e-12
            if o.agent_name == "Diversification":
                assert (v[:-1] <= ctx.max_weights + 1e-12).all(), pos
            assert np.isfinite(v).all() and o.agent_name in AGENTS


def test_liquidity_invalid_asset_is_excluded_and_the_eligible_set_recovers(env):
    out = {pos: {o.agent_name: o for o in run_all(at(env, pos)[2], env["cfg"])} for pos in (27, 28, 30)}
    for name in AGENTS:
        assert out[28][name].normalized_score["NYSE_C"] == 0.0, name  # flagged liquidity_invalid on 2024-03-14
    d = out[28]["Diversification"].diagnostics
    assert d["eligible"] == [True, True, False] and d["B_div_effective"] == pytest.approx(0.8 * 2 / 3)
    assert out[27]["Diversification"].diagnostics["eligible"] == [True, True, True]
    assert out[30]["Diversification"].diagnostics["eligible"] == [True, True, True]  # temporary invalidation ended


def test_the_cap_active_diversification_case_occurs_on_the_fixture(env):
    d = diversification.run(at(env, 31)[2], env["cfg"])
    caps = np.array(d.diagnostics["sleeve_caps"])
    assert d.normalized_score["KRX_B"] == pytest.approx(caps[1], abs=1e-6) == pytest.approx(0.75, abs=1e-6)
    assert d.absolute_target["KRX_B"] <= 0.6 + 1e-12 and d.diagnostics["agent_status"] == "ok"
    assert d.diagnostics["solver"]["attempts"][0]["status"] == "optimal" and not d.diagnostics.get("fallback_used")


def test_defensive_and_diversification_are_similar_but_not_identical_on_the_fixture(env):
    cos = []
    for pos in (10, 25, 31, 41, 50, 60, 70):
        ctx = at(env, pos)[2]
        a, b = defensive.run(ctx, env["cfg"]), diversification.run(ctx, env["cfg"])
        x, y = np.array(list(a.normalized_score.values())), np.array(list(b.normalized_score.values()))
        cos.append(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)))
        assert not np.allclose(x, y, atol=1e-6) and a.diagnostics.keys() != b.diagnostics.keys()
    assert 0.5 < np.median(cos) <= 1.0 and max(cos) < 1.0  # reported diagnostic only; threshold is OQ-018


def test_fixed_regime_table_probabilities_follow_the_scenario_blocks(env):
    for pos in LOW_POSITIONS:
        assert env["table"].asof(at(env, pos)[0])[1] == "low_risk", pos
    for pos in HIGH_POSITIONS:
        assert env["table"].asof(at(env, pos)[0])[1] == "high_risk", pos


def mixture(env, pos):
    cut, fit, ctx = at(env, pos)
    cfg = env["cfg"]
    probs, label = env["table"].asof(cut)
    comps = {k: v[0] for k, v in pressure_components(env, ctx, fit).items()}
    gate = next(g for g in cfg.regime.agent_gate_candidates if g.name == "fixture_gate")
    d = cfg.agents.defensive
    c = combine(run_all(ctx, cfg), ctx.assets, probs, ["low_risk", "high_risk"], gate.by_state_label,
                p_high=float(probs[1]), pressure_components=comps, base=d.overlay_base_risky_budget,
                minimum=d.overlay_min_risky_budget, gamma=d.overlay_gamma_candidates[1],
                target_cash_floor=cfg.constraints.target_cash_floor, max_weights=ctx.max_weights,
                usd_mask=np.array([a.currency == "USD" for a in cfg.universe.assets]), usd_limit=cfg.constraints.usd_exposure_limit)
    return c, label, probs


def test_defensive_overlay_raises_cash_in_the_high_risk_fixture_state(env):
    low = [mixture(env, p)[0] for p in LOW_POSITIONS]
    high = [mixture(env, p)[0] for p in HIGH_POSITIONS]
    d = env["cfg"].agents.defensive
    # p_high = 0.9 in the high blocks: B_defensive = 0.2 + 0.6 * (1 - pressure)^1 <= 0.26
    for c in high:
        assert c.b_defensive <= d.overlay_min_risky_budget + 0.6 * 0.1 + 1e-9 and c.a_bar[CASH] >= 0.74 - 1e-9
    assert np.mean([c.a_bar[CASH] for c in high]) > np.mean([c.a_bar[CASH] for c in low])
    assert all(c.b_final <= c.b_defensive + 1e-12 for c in low + high)


def test_no_risky_only_renormalization_on_the_fixture(env):
    for p in LOW_POSITIONS + HIGH_POSITIONS:
        c, _, _ = mixture(env, p)
        gross = sum(c.a_bar[a] for a in ("KRX_A", "KRX_B", "NYSE_C"))
        assert gross <= c.b_final + 1e-12 and c.a_bar[CASH] == pytest.approx(1 - gross, abs=1e-12)
        assert c.b_final <= min(c.b_proposed, c.b_defensive, 1 - env["cfg"].constraints.target_cash_floor) + 1e-12
        assert sum(c.a_bar.values()) == pytest.approx(1.0, abs=1e-12) and min(c.a_bar.values()) >= -1e-12
        assert all(c.a_bar[a] <= 0.6 + 1e-12 for a in ("KRX_A", "KRX_B", "NYSE_C"))  # caps hold after the mixture


def test_pressure_thresholds_are_fitted_on_training_rows_only(env):
    _, fit, ctx = at(env, 40)
    comps = pressure_components(env, ctx, fit)
    for name, (p, th) in comps.items():
        assert 0.0 <= p <= 1.0 and th.n_training <= ctx.train_end_row - ctx.train_start_row + 1, name
    assert comps["first_pca_variance_share"][1].reason in {"fitted", "insufficient_spread", "insufficient_training_decisions"}
    # thresholds computed at a later fit use more training rows, proving they are re-fit per fold and not copied backward
    later = pressure_components(env, at(env, 60)[2], at(env, 60)[1])
    assert later["market_drawdown_magnitude"][1].n_training > comps["market_drawdown_magnitude"][1].n_training


def test_decision_outputs_are_invariant_to_data_after_the_cutoff(env, tmp_path):
    cut, _, ctx = at(env, 41)
    root = tmp_path / "p"
    shutil.copytree(ROOT / "data" / "fixtures", root / "data" / "fixtures")
    for name in ("prices_raw", "prices_total_return", "fx_krw_per_usd"):
        p = root / "data" / "fixtures" / "fixture_3asset" / f"{name}.parquet"
        df = pd.read_parquet(p)
        late = pd.to_datetime(df.available_at_utc, utc=True) > cut
        df.loc[late, "value"] *= 1.2
        df.to_parquet(p, index=False)
    md2 = load_market_data(env["cfg"], root)
    _, _, ctx2 = at(env, 41, md=md2)
    for a, b in zip(run_all(ctx, env["cfg"]), run_all(ctx2, env["cfg"]), strict=True):
        assert a.absolute_target == b.absolute_target and a.confidence == b.confidence, a.agent_name


def test_em_hmm_pipeline_runs_on_fixture_features_and_is_degenerate_free(env):
    _, fit, ctx = at(env, 40)
    feats = hmm_features(fit, ctx.returns)
    h = env["cfg"].regime.hmm
    r = run_hmm_pipeline(feats, None, n_states=2, seed=env["cfg"].reproducibility.random_streams.hmm_em,
                         max_iter=h.max_iterations, tol=h.em_tolerance, feature_order=HMM_FEATURE_ORDER)
    assert r.raw.converged and np.isfinite(r.train.filtered).all()
    np.testing.assert_allclose(r.train.filtered.sum(axis=1), 1.0, atol=1e-12)
    vol = np.sqrt(np.array(r.aligned.variances).mean(axis=1))
    assert vol[0] < vol[1]  # state 0 = low risk after alignment
