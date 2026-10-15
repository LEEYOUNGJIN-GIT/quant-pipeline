"""Phase 4 gate (spec 19) on the real fixture: scale lineage is train-only; objective contributions reconcile; plan hashes
are stable; execution shortfall is explicit. Evidence class: fixture_path."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.calendars import Calendars, build_row_map
from quant_pipeline.config import load_config
from quant_pipeline.data import load_initial_holdings, load_market_data
from quant_pipeline.decision import FixedRegimeSource, TrialParams, calibrate_fold, decide
from quant_pipeline.fitstate import agent_context, decision_cutoff, fit_core, lineage_hashes
from quant_pipeline.objective import expected_cost_rates, objective_terms
from quant_pipeline.optimizer import OptimizerInputs, hard_constraint_violations, optimize
from quant_pipeline.rebalance import LedgerSnapshot, build_plan
from quant_pipeline.regimes import FixedRegimeTable

ROOT = Path(__file__).resolve().parents[2]
TRAIN_POSITION = 40


@pytest.fixture(scope="module")
def env():
    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    cals = Calendars.load(["XKRX", "XNYS"])
    rm, sess = build_row_map(cfg, cals)
    return {"cfg": cfg, "cals": cals, "rm": rm, "sess": sess, "md": load_market_data(cfg, ROOT), "hashes": lineage_hashes(cfg, ROOT),
            "regime": FixedRegimeSource(FixedRegimeTable.load(ROOT / cfg.regime.fixed_table.path)),
            "usd": np.array([a.currency == "USD" for a in cfg.universe.assets])}


def fold(env, md=None, position=TRAIN_POSITION):
    md = md or env["md"]
    cfg, cals, rm = env["cfg"], env["cals"], env["rm"]
    fit = fit_core(cfg, md, cals, env["sess"], fold_id="outer-1", refit_id="r0", hashes=env["hashes"],
                   fit_end_utc=decision_cutoff(cals, cfg, rm.eligible[position - 1]))
    cuts = [decision_cutoff(cals, cfg, s) for s in rm.eligible[:position]]
    cal = calibrate_fold(cfg, md, cals, env["sess"], fit, env["regime"], TrialParams.first(cfg), cuts, np.array([0.0, 0.0, 0.0, 1.0]))
    return fit, cal, cuts


@pytest.fixture(scope="module")
def base(env):
    return fold(env)


def perturbed(env, tmp_path: Path, after: pd.Timestamp | None = None, before: pd.Timestamp | None = None):
    root = tmp_path / ("future" if after is not None else "past")
    shutil.copytree(ROOT / "data" / "fixtures", root / "data" / "fixtures")
    for name in ("prices_raw", "prices_total_return", "fx_krw_per_usd"):
        p = root / "data" / "fixtures" / "fixture_3asset" / f"{name}.parquet"
        df = pd.read_parquet(p)
        av = pd.to_datetime(df.available_at_utc, utc=True)
        mask = (av > after) if after is not None else (av <= before) & (av > pd.Timestamp("2024-02-15T00:00:00Z"))
        df.loc[mask, "value"] = df.loc[mask, "value"] * (1.0 + 0.04 * np.cos(np.arange(mask.sum())))
        df.to_parquet(p, index=False)
    return load_market_data(env["cfg"], root)


def test_calibration_uses_exactly_the_forty_training_decisions(base):
    _, cal, cuts = base
    assert cal.n_decisions == len(cuts) == 40 and cal.w_track.sum() == pytest.approx(4.0)
    assert cal.v_minus.shape == cal.a_bar.shape == (40, 4)
    np.testing.assert_allclose(cal.v_minus[0], [0, 0, 0, 1.0])  # first virtual pretrade row is the marked initial holdings
    np.testing.assert_allclose(cal.v_minus.sum(axis=1), 1.0, atol=1e-12)
    for s in (cal.s_track, cal.s_risk, cal.s_cost):
        assert not s.fallback and s.n_positive == 40 and s.value >= s.floor and list(s.sorted_sample) == sorted(s.sorted_sample)


def test_scale_lineage_is_train_only_future_data_cannot_change_calibration(env, base, tmp_path):
    fit, cal, _ = base
    md2 = perturbed(env, tmp_path, after=fit.fit_end_utc)
    _, cal2, _ = fold(env, md=md2)
    np.testing.assert_array_equal(cal.w_track, cal2.w_track)
    assert cal.scales == cal2.scales and np.array_equal(cal.a_bar, cal2.a_bar) and np.array_equal(cal.v_minus, cal2.v_minus)


def test_calibration_does_react_to_training_data_so_the_invariance_is_not_vacuous(env, base, tmp_path):
    fit, cal, _ = base
    md2 = perturbed(env, tmp_path, before=fit.fit_end_utc)
    _, cal2, _ = fold(env, md=md2)
    assert cal.scales != cal2.scales or not np.array_equal(cal.a_bar, cal2.a_bar)


def test_a_calibration_decision_outside_the_training_window_is_refused(env, base):
    fit, _, _ = base
    late = decision_cutoff(env["cals"], env["cfg"], env["rm"].eligible[TRAIN_POSITION])  # position 41 is OOS
    with pytest.raises(ValueError, match="training window"):
        calibrate_fold(env["cfg"], env["md"], env["cals"], env["sess"], fit, env["regime"], TrialParams.first(env["cfg"]), [late],
                       np.array([0.0, 0.0, 0.0, 1.0]))


def decision_inputs(env, fit, cal, position, w_minus, lambdas=(1.0, 1.0, 1.0)):
    cut = decision_cutoff(env["cals"], env["cfg"], env["rm"].eligible[position - 1])
    ctx = agent_context(env["cfg"], env["md"], env["cals"], env["sess"], fit, cut)
    tgt = decide(env["cfg"], ctx, env["regime"], TrialParams.first(env["cfg"]), cut)
    inp = OptimizerInputs(tgt.a_bar, np.asarray(w_minus, float), fit.covariance.matrix, cal.w_track, cal.scales, lambdas,
                          expected_cost_rates(env["cfg"]), tgt.combined.b_final, env["cfg"].constraints.target_cash_floor,
                          ctx.max_weights, env["usd"], env["cfg"].constraints.usd_exposure_limit,
                          env["cfg"].constraints.turnover_one_way_limit, ctx.tradable)
    return cut, ctx, tgt, inp


def test_objective_contributions_reconcile_on_fixture_decisions(env, base):
    fit, cal, _ = base
    w_minus = np.array([0.0, 0.0, 0.0, 1.0])
    for pos in range(41, 51):
        _, _, tgt, inp = decision_inputs(env, fit, cal, pos, w_minus)
        r = optimize(inp, env["cfg"])
        assert r.branch in ("optimal", "numerical_retry_optimal") and not r.halted, pos
        t = objective_terms(r.weights, tgt.a_bar, w_minus, cal.w_track, fit.covariance.matrix, inp.rates, cal.scales, inp.lambdas)
        assert sum(t.contributions) == pytest.approx(r.terms.objective, rel=1e-12) and all(c >= 0 for c in t.contributions)
        assert max(hard_constraint_violations(r.weights, inp).values()) <= 1e-8
        assert r.weights[:3].sum() <= tgt.combined.b_final + 1e-8 and r.weights[3] >= 0.05 - 1e-8
        w_minus = r.weights  # carry the intention forward (a stand-in for the realized state in Phase 5)


def test_the_optimizer_beats_the_naive_candidates_it_could_have_chosen(env, base):
    fit, cal, _ = base
    _, _, tgt, inp = decision_inputs(env, fit, cal, 41, [0.0, 0.0, 0.0, 1.0])
    r = optimize(inp, env["cfg"])
    for cand in (inp.w_minus, tgt.a_bar):
        if max(hard_constraint_violations(cand, inp).values()) <= 1e-9:  # only compare against feasible candidates
            c = objective_terms(cand, tgt.a_bar, inp.w_minus, cal.w_track, fit.covariance.matrix, inp.rates, cal.scales, inp.lambdas)
            assert r.terms.objective <= c.objective + 1e-9


def end_to_end(env, position=41):
    fit, cal, _ = fold(env)
    cut, _, _, inp = decision_inputs(env, fit, cal, position, [0.0, 0.0, 0.0, 1.0])
    r = optimize(inp, env["cfg"])
    cash, qty, asof = load_initial_holdings(env["cfg"], ROOT)
    plan = build_plan(env["cfg"], env["md"], env["cals"], LedgerSnapshot(cash, qty, pd.Timestamp(asof)), r.weights, run_id="run",
                      fold_id="outer-1", refit_id="r0", decision_id=f"D{position:03d}", trial_hash="sha256:trial", decision_cutoff=cut)
    return r, plan


def test_plan_hash_is_stable_across_runs_and_across_processes(env):
    r1, p1 = end_to_end(env)
    r2, p2 = end_to_end(env)
    assert p1.plan_content_hash == p2.plan_content_hash and np.array_equal(r1.weights, r2.weights)
    code = ("import sys, json; sys.path.insert(0, 'tests/integration'); import test_phase4_fixture as t;"
            "env = t.env.__wrapped__(); r, p = t.end_to_end(env); print(json.dumps(p.plan_content_hash))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == p1.plan_content_hash  # a fresh interpreter reproduces it


def test_execution_shortfall_is_explicit_in_the_plan(env):
    _, p = end_to_end(env)
    gross_target, gross_intended = p.summary["target_risky_gross"], p.summary["intended_risky_gross"]
    assert 0 <= gross_target - gross_intended < 0.001  # only lot rounding toward zero separates intention from holdings
    for o in p.orders:  # every order reports what was wanted, what is planned and why they differ
        assert o.unconstrained_trade_quantity >= o.intended_trade_quantity >= 0 and o.shortfall_quantity == 0
    poor = LedgerSnapshot(Decimal("1000000000.00"), {"KRX_A": 0, "KRX_B": 0, "NYSE_C": 0}, pd.Timestamp("2024-04-03T21:15:00Z"))
    tight = build_plan(env["cfg"], env["md"], env["cals"], poor, np.array([0.40, 0.30, 0.2999, 0.0001]), run_id="run", fold_id="f",
                       refit_id="r", decision_id="D", trial_hash="t", decision_cutoff=pd.Timestamp("2024-04-03T21:15:00Z"))
    assert tight.summary["shortfall_orders"] and Decimal(tight.projected_post_cost_cash_krw) >= 0
    assert all(o.shortfall_reason == "cash_feasibility" for o in tight.orders if o.shortfall_quantity)


def test_the_plan_never_contains_fills_or_prices_from_after_the_decision(env):
    _, p = end_to_end(env)
    cut = pd.Timestamp(p.decision_cutoff_utc)
    for o in p.orders:
        asset_price = env["md"].raw_mark(o.asset_id, cut)  # the sizing price is the as-of mark at the decision cutoff
        assert Decimal(o.sizing_price_raw) == asset_price and pd.Timestamp(o.intended_execution_utc) > cut
