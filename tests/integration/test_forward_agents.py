"""Agent-level view of a forward path (tools/forward_agents.py) on a SYNTHETIC snapshot: engineering evidence, never a research result."""
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
spec = importlib.util.spec_from_file_location("forward_agents_tool", ROOT / "tools" / "forward_agents.py")
G = importlib.util.module_from_spec(spec)
sys.modules["forward_agents_tool"] = G
spec.loader.exec_module(G)
from support.research_smoke import write_smoke_config
from support.synthetic_research import build_research_repo

START = "2022-01-03"


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    root, _ = build_research_repo(tmp_path_factory.mktemp("fwd_agents"), start="2020-01-02", end="2022-12-30")
    p = G.ForwardPipeline(root, root / "out", "g1", config=write_smoke_config(root), allow_synthetic=True)
    t = p.ev.design.trials[0]
    return G.agent_view(p, G.Frozen(t["trial_hash"], t["parameters"]), START)


def test_shares_renormalise_the_risky_vector_and_ignore_an_empty_one():
    s = G.shares(np.array([0.2, 0.2, 0.1]), ["005930.KS", "000660.KS", "IEF"])
    assert s == {"korean_equity": 0.8, "treasury_etf": 0.2}
    assert G.shares(np.zeros(3), ["a", "b", "c"]) is None


def test_the_binding_term_is_the_smallest_cap_and_ties_are_named_together():
    assert G.binding_term(0.7, 0.4, 0.95, 0.4) == "defensive_envelope"
    assert G.binding_term(0.4, 0.4, 0.95, 0.4) == "proposed+defensive_envelope"
    assert G.binding_term(0.0, 0.0, 0.95, 0.0) == "proposed+defensive_envelope"
    assert G.binding_term(0.7, 0.9, 0.95, 0.0) == "zero_mixture"


def test_the_view_covers_every_decision_and_the_four_agents(view):
    assert view["summary"]["decisions"] >= 2 and len(view["agent_order"]) == 4
    for r in view["decisions"]:
        assert set(r["agents"]) == set(view["agent_order"]) and r["b_final"] <= min(r["b_proposed"], r["b_defensive"]) + 1e-9
        assert sum(r["nav_share_by_group"]["target_before_optimiser"].values()) == pytest.approx(r["b_final"] - r["cap_reduction"], abs=2e-4)
        assert r["optimiser_shift_half_l1"] >= 0.0


def test_the_summary_passes_the_wording_rules_and_holds_no_levels(view):
    text = json.dumps(view)
    assert not G.forbidden_in(text) and "nav_krw" not in text and "qty_" not in text
