"""Optimiser view of a forward path (tools/forward_optimiser.py) on a SYNTHETIC snapshot: engineering evidence, never a research result."""
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
spec = importlib.util.spec_from_file_location("forward_optimiser_tool", ROOT / "tools" / "forward_optimiser.py")
O = importlib.util.module_from_spec(spec)
sys.modules["forward_optimiser_tool"] = O
spec.loader.exec_module(O)
from support.research_smoke import write_smoke_config
from support.synthetic_research import build_research_repo

START = "2022-01-03"


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    root, _ = build_research_repo(tmp_path_factory.mktemp("fwd_opt"), start="2020-01-02", end="2022-12-30")
    p = O.ForwardPipeline(root, root / "out", "o1", config=write_smoke_config(root), allow_synthetic=True)
    t = p.ev.design.trials[0]
    return O.optimiser_view(p, O.Frozen(t["trial_hash"], t["parameters"]), START)


def test_risk_shares_add_to_one_and_a_diagonal_model_gives_each_asset_its_variance_share():
    sigma = np.diag([0.04, 0.01, 0.01])
    out = O.risk_shares(np.array([0.5, 0.5, 0.0]), sigma, ["005930.KS", "IEF", "TLT"])
    assert out["korean_equity"] == pytest.approx(0.04 * 0.25 / (0.04 * 0.25 + 0.01 * 0.25), abs=1e-4)
    assert sum(out.values()) == pytest.approx(1.0, abs=2e-4)
    assert O.risk_shares(np.zeros(3), sigma, ["a", "b", "c"]) is None


def test_volatility_ratios_are_relative_to_the_median_asset():
    sigma = np.diag([0.04, 0.01, 0.01])
    r = O.volatility_ratios(sigma, ["005930.KS", "IEF", "TLT"])
    assert r == {"korean_equity": 2.0, "treasury_etf": 1.0}


def test_the_sleeve_volatility_is_per_unit_of_risky_exposure():
    sigma = np.diag([0.04, 0.04])
    assert O.sleeve_volatility(np.array([0.1, 0.0]), sigma) == pytest.approx(0.2)
    assert O.sleeve_volatility(np.zeros(2), sigma) is None


def test_the_reconstructed_inputs_reproduce_the_recorded_weights(view):
    assert view["summary"]["reconstruction_max_abs_weight_difference_at_selected"] < 1e-5


def test_every_scenario_is_solved_for_every_decision_and_the_selected_one_matches_the_record(view):
    n = view["summary"]["decisions"]
    assert n >= 2 and view["sensitivity"]["selected"]["failed"] == 0
    for name, s in view["sensitivity"].items():
        assert s["solves"] == n and s["failed"] <= n, name
    assert view["sensitivity"]["selected"]["mean_shift_from_target_half_l1"] >= 0.0


def test_the_summary_passes_the_wording_rules_and_holds_no_levels(view):
    text = json.dumps(view)
    assert not O.forbidden_in(text) and "nav_krw" not in text and "qty_" not in text
