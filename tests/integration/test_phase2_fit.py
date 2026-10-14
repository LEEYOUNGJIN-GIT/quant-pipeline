"""Phase 2 gate (spec 19): no global fit; future perturbations preserve prior artifacts; Ledoit-Wolf present.
Evidence class: fixture_path (real fixture panel through the production fold-local fit)."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.calendars import Calendars, build_row_map
from quant_pipeline.config import load_config
from quant_pipeline.data import load_market_data
from quant_pipeline.dependency import cache_key
from quant_pipeline.fitstate import decision_cutoff, fit_core, lineage_hashes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def env():
    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    cals = Calendars.load(["XKRX", "XNYS"])
    rm, sess = build_row_map(cfg, cals)
    return {"cfg": cfg, "cals": cals, "rm": rm, "sess": sess, "md": load_market_data(cfg, ROOT),
            "hashes": lineage_hashes(cfg, ROOT)}


def fit(env, position: int, md=None, fold="outer-1", refit="r0", **kw):
    end = decision_cutoff(env["cals"], env["cfg"], env["rm"].eligible[position - 1])
    return fit_core(env["cfg"], md or env["md"], env["cals"], env["sess"], fold_id=fold, refit_id=refit, fit_end_utc=end,
                    hashes=env["hashes"], **kw)


def perturbed(env, tmp_path: Path, after: pd.Timestamp, inside_only: bool = False):
    root = tmp_path / ("in" if inside_only else "out")
    shutil.copytree(ROOT / "data" / "fixtures", root / "data" / "fixtures")
    for name in ("prices_raw", "prices_total_return"):
        p = root / "data" / "fixtures" / "fixture_3asset" / f"{name}.parquet"
        df = pd.read_parquet(p)
        av = pd.to_datetime(df.available_at_utc, utc=True)
        mask = (av <= after) if inside_only else (av > after)
        df.loc[mask, "value"] = df.loc[mask, "value"] * (1.0 + 0.03 * np.sin(np.arange(mask.sum())))
        df.to_parquet(p, index=False)
    return load_market_data(env["cfg"], root)


def same_artifacts(a, b) -> bool:
    return (np.array_equal(a.pca.standardizer.mean, b.pca.standardizer.mean)
            and np.array_equal(a.pca.standardizer.scale, b.pca.standardizer.scale)
            and np.array_equal(a.pca.loadings, b.pca.loadings) and np.array_equal(a.pca.residuals, b.pca.residuals)
            and np.array_equal(a.covariance.matrix, b.covariance.matrix) and a.covariance.diagnostics == b.covariance.diagnostics
            and a.asynchrony == b.asynchrony and a.key == b.key)


def test_fit_geometry_for_the_first_outer_fold(env):
    f = fit(env, 40)
    assert len(f.panel[f.panel.complete_case]) == 20 + 40  # warmup + the 40 initial training rows
    assert f.covariance.n_observations == env["cfg"].covariance.lookback_rows == 20
    assert f.fit_end_utc <= decision_cutoff(env["cals"], env["cfg"], env["rm"].eligible[39])
    assert f.covariance.estimator_name.startswith("LedoitWolf") and 0 <= f.covariance.diagnostics["shrinkage"] <= 1


def test_fit_is_deterministic(env):
    assert same_artifacts(fit(env, 40), fit(env, 40))


def test_there_is_no_global_fit_each_fold_has_its_own_standardizer(env):
    a, b = fit(env, 40), fit(env, 50)
    assert not np.array_equal(a.pca.standardizer.mean, b.pca.standardizer.mean)
    assert b.pca.standardizer.n_observations == a.pca.standardizer.n_observations + 10
    assert a.key != b.key and not np.array_equal(a.covariance.matrix, b.covariance.matrix)


def test_future_perturbation_cannot_change_any_fold_artifact(env, tmp_path):
    base = fit(env, 40)
    md2 = perturbed(env, tmp_path, after=base.fit_end_utc)
    other = fit(env, 40, md=md2)
    assert same_artifacts(base, other)  # means, scales, loadings, residuals, covariance, diagnostics, cache key
    later_a, later_b = fit(env, 60), fit(env, 60, md=md2)
    assert not same_artifacts(later_a, later_b)  # the perturbation is real: a later fit that sees it does change


def test_perturbing_data_inside_the_window_changes_the_key(env, tmp_path):
    base = fit(env, 40)
    md2 = perturbed(env, tmp_path, after=base.fit_end_utc, inside_only=True)
    assert fit(env, 40, md=md2).key != base.key


def test_fold_panel_is_exactly_the_prefix_of_the_oracle_panel(env):
    f = fit(env, 40)
    oracle = pd.read_csv(ROOT / "tests/expected/fixture_3asset/oracle/panel_daily_weekly.csv")
    want = oracle[(oracle.kind == "daily") & (pd.to_datetime(oracle.cutoff_utc, utc=True) <= f.fit_end_utc)]
    pd.testing.assert_frame_equal(f.panel.reset_index(drop=True), want.reset_index(drop=True), check_dtype=False,
                                  rtol=1e-12, atol=1e-12)


def test_lineage_ids_do_not_enter_the_cache_key_but_inputs_do(env):
    a = fit(env, 40, fold="outer-1", refit="r0")
    b = fit(env, 40, fold="some-other-fold", refit="r9")
    assert a.key == b.key and a.fold_id != b.fold_id  # identical information -> shared key
    assert fit(env, 40, n_components=2).key != a.key  # a relevant parameter changes it
    assert fit(env, 41).key != a.key  # a later information cutoff changes it
    bad = copy.deepcopy(a.fingerprint)
    bad["fold_id"] = "x"
    with pytest.raises(ValueError, match="lineage"):
        cache_key(bad)


def test_fingerprint_is_truthful_about_the_missing_lock_and_trial_influence(env):
    fp = fit(env, 40).fingerprint
    assert fp["lock_hash"] == "unavailable:no_accepted_lock"
    assert fp["trial_influence"]["influenced"] is False and fp["trial_influence"]["excluded_dimensions"]
    assert fp["base_profile_hash"].startswith("sha256:") and fp["asset_order"] == ["KRX_A", "KRX_B", "NYSE_C"]


def test_asynchrony_diagnostics_are_populated_and_honest(env):
    f = fit(env, 40)
    d = f.asynchrony
    cov_rows = int(f.panel.complete_case.sum())
    assert d["simultaneous_update_rows"] == cov_rows
    assert d["pairwise_valid_overlap"]["KRX_A|NYSE_C"] >= cov_rows
    zero = d["coverage"]["per_asset"]["KRX_A"]["true_zero_returns"]
    assert zero == 1  # the genuine unchanged KRX_A observation (2024-02-27) is a real zero, kept distinct from nulls
    per = d["coverage"]["per_asset"]
    assert per["NYSE_C"]["no_update_fraction"] == 0.0  # XNYS is the anchor calendar: every row has a new NYSE observation
    assert per["KRX_A"]["no_update_fraction"] > 0.0  # XKRX holidays leave KRX assets un-updated (null, not zero)
    assert per["KRX_A"]["age_hours"]["max"] > per["NYSE_C"]["age_hours"]["max"]  # carried KRX levels are older
    assert set(d["cross_correlations"]["KRX_A|NYSE_C"]) == {"-2", "-1", "0", "1", "2"}
    assert d["weekly_common_cutoff"]["status"] == "ok" and d["alternate_anchor_xkrx"]["status"] == "ok"
    for key in ("weekly_common_cutoff", "alternate_anchor_xkrx"):
        cmp = d[key]["optimal_weights_comparison"]
        assert abs(sum(cmp["daily_baseline_weights"]) - 1) < 1e-9 and abs(sum(cmp["other_weights"]) - 1) < 1e-9
        assert min(cmp["daily_baseline_weights"] + cmp["other_weights"]) >= 0 and 0 <= cmp["one_way_turnover_between_them"] <= 1
    assert "approximation" in d["limitation"]


def test_weekly_sensitivity_reports_insufficient_history_instead_of_guessing(env):
    d = fit(env, 1).asynchrony["weekly_common_cutoff"]
    assert d["status"] == "insufficient_weekly_history" and d["minimum_rows"] == 8 and d["complete_case_rows"] < 8
