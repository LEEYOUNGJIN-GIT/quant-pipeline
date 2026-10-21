"""Descriptive decomposition of a forward path (tools/forward_decompose.py) on a SYNTHETIC snapshot: engineering evidence, never a research result."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("forward_decompose_tool", ROOT / "tools" / "forward_decompose.py")
D = importlib.util.module_from_spec(spec)
sys.modules["forward_decompose_tool"] = D
spec.loader.exec_module(D)
from support.research_smoke import write_smoke_config
from support.synthetic_research import build_research_repo

START = "2022-01-03"


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    root, _ = build_research_repo(tmp_path_factory.mktemp("fwd_dec"), start="2020-01-02", end="2022-12-30")
    p = D.ForwardPipeline(root, root / "out", "d1", config=write_smoke_config(root), allow_synthetic=True)
    t = p.ev.design.trials[0]
    return D.decompose(p, D.Frozen(t["trial_hash"], t["parameters"]), START)


def test_matching_the_benchmarks_own_exposure_returns_the_benchmark():
    bench, rf = np.array([0.01, -0.02, 0.03]), np.array([0.001, 0.001, 0.001])
    full = np.ones(3)
    assert np.allclose(D.exposure_matched(bench, rf, full, full), bench)
    assert np.allclose(D.exposure_matched(bench, rf, np.zeros(3), full), rf), "no exposure earns the risk-free rate"
    assert np.allclose(D.exposure_matched(bench, rf, 0.5 * full, full), 0.5 * bench + 0.5 * rf)


def test_the_worst_drawdown_episode_is_found():
    r = np.array([0.1, -0.2, -0.1, 0.05, 0.3])
    peak, trough, depth = D.drawdown_episode(r)
    assert (peak, trough) == (1, 2) and depth == pytest.approx(1 - 0.8 * 0.9)


def test_the_summary_is_complete_and_uses_only_the_scored_intervals(out):
    assert out["first_session"] >= START and out["intervals"] > 20
    assert set(out["paths"]) == {"strategy", D.PRIMARY_BENCHMARK, "benchmark_at_strategy_exposure_each_interval", "benchmark_at_strategy_mean_exposure"}
    for k in ("strategy", D.PRIMARY_BENCHMARK):
        e = out["risky_share"][k]
        assert 0.0 <= e["p10"] <= e["median"] <= e["p90"] <= 1.0 + 1e-9 and 0.0 <= e["mean"] <= 1.0 + 1e-9
    assert out["decisions"]["count"] >= 2 and out["missing_marks"] == {"strategy": 0, D.PRIMARY_BENCHMARK: 0}


def test_the_attribution_tracks_the_interval_returns_it_approximates(out):
    for k, c in out["contribution_sum_vs_arithmetic_return"].items():
        assert abs(c["attributed"] - c["interval_returns_sum"]) < 0.5 * max(abs(c["interval_returns_sum"]), 0.05) + 0.05, k


def test_the_summary_passes_the_wording_rules_and_holds_no_levels(out):
    text = json.dumps(out)
    assert not D.forbidden_in(text) and "nav_krw" not in text and "qty_" not in text


def test_the_exposure_and_selection_effects_add_up_to_the_gap_when_the_paths_are_built_from_sleeves():
    rng = np.random.default_rng(7)
    n, k = 40, 4
    r = rng.normal(0.0005, 0.01, (n, k))
    ws = rng.dirichlet(np.ones(k + 1), n)[:, :k]
    wb = np.tile(np.array([0.2375] * k), (n, 1))
    rf = np.full(n, 0.0001)
    rs, s = D.sleeve_return(ws, r)
    rb, b = D.sleeve_return(wb, r)
    arr_s, arr_b = s * rs + (1 - s) * rf, b * rb + (1 - b) * rf
    out = D.sleeve_analysis([f"A{i}" for i in range(k)], ws, wb, r, rf, arr_s, arr_b, np.array(["2025Q1"] * n), [f"2025-01-{i + 1:02d}" for i in range(n)], [], 252)
    assert abs(out["residual"]) < 1e-3 and out["sum_of_gap"] == pytest.approx(out["exposure_effect"] + out["selection_effect"], abs=1e-3)


def test_dropping_an_asset_renormalises_the_sleeve_without_it():
    w = np.array([[0.2, 0.3, 0.1]])
    r = np.array([[0.10, 0.00, -0.10]])
    full, share = D.sleeve_return(w, r)
    dropped, share_d = D.sleeve_return(w, r, drop=0)
    assert share[0] == pytest.approx(0.6) and full[0] == pytest.approx((0.02 - 0.01) / 0.6)
    assert share_d[0] == pytest.approx(0.4) and dropped[0] == pytest.approx(-0.01 / 0.4)


def test_the_sleeve_analysis_is_complete_and_its_decision_table_covers_the_decisions(out):
    sl = out["sleeve_analysis"]
    assert set(sl) >= {"exposure_effect", "selection_effect", "residual", "by_group", "leave_one_asset_out", "decision_table", "by_quarter"}
    assert len(sl["decision_table"]) == out["decisions"]["count"] and all(0.0 <= d["risky_budget"] <= 1.0 for d in sl["decision_table"])
    assert sum(g["mean_nav_share"]["strategy"] for g in sl["by_group"].values()) == pytest.approx(out["risky_share"]["strategy"]["mean"], abs=1e-3)
