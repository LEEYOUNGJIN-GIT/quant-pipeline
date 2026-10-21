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
