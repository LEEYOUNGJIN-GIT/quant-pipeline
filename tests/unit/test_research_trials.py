"""Research trial space (spec 11.3, OQ-043): cardinality 5832 with PCA component count and gate table as trial dimensions; fixture frozen."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from quant_pipeline.config import load_config
from quant_pipeline.trials import enumerate_trials, to_trial_params, trial_dimensions

RES = load_config(ROOT / "configs" / "research_8_12asset.yaml")
FIX = load_config(ROOT / "configs" / "fixture_3asset_aux1024.yaml")


def test_the_research_cardinality_is_the_specs_5832():
    d = enumerate_trials(RES)
    assert d.cardinality == 5832 == 486 * 3 * 4 and len(d.trials) == RES.evaluation.max_parameter_trials == 64
    assert {"pca.n_components_candidates", "regime.agent_gate_candidates"} <= set(d.dimension_order)
    assert d.stop_reason == "cap_reached"


def test_every_research_trial_names_a_declared_component_count_and_gate():
    d = enumerate_trials(RES)
    gates = {g.name for g in RES.regime.agent_gate_candidates}
    for t in d.trials:
        assert t["parameters"]["pca.n_components_candidates"] in RES.pca.n_components_candidates
        assert t["parameters"]["regime.agent_gate_candidates"] in gates
    assert len({t["parameters"]["pca.n_components_candidates"] for t in d.trials}) == 3  # the 64 draws cover all three counts
    assert len({t["parameters"]["regime.agent_gate_candidates"] for t in d.trials}) == 4


def test_the_fixture_trial_space_and_hashes_are_untouched():
    d = enumerate_trials(FIX)
    assert d.cardinality == 8 and set(d.trials[0]["parameters"]) == set(trial_dimensions(FIX)) and len(trial_dimensions(FIX)) == 6
    assert "pca.n_components_candidates" not in d.trials[0]["parameters"]
    assert d.design_hash == "sha256:35a63bfd5c72e77aaa100f20e257fce1ff13062d4f698344e6da923c35a447d9"  # identical before and after this change


def test_trial_parameters_map_to_the_gate_and_component_count():
    t = enumerate_trials(RES).trials[0]
    p = to_trial_params(RES, t["parameters"])
    assert p.gate_name == t["parameters"]["regime.agent_gate_candidates"] and p.n_components == t["parameters"]["pca.n_components_candidates"]
    fx = to_trial_params(FIX, enumerate_trials(FIX).trials[0]["parameters"])
    assert fx.n_components is None and fx.gate_name == FIX.regime.agent_gate_candidates[0].name


def test_different_gate_or_component_choices_are_different_trial_hashes():
    d = enumerate_trials(RES)
    assert len({t["trial_hash"] for t in d.trials}) == len(d.trials)
    for dim in ("pca.n_components_candidates", "regime.agent_gate_candidates"):
        a = next(t for t in d.trials)
        b = next(t for t in d.trials if t["parameters"][dim] != a["parameters"][dim])
        from quant_pipeline.trials import trial_hash
        assert trial_hash({**a["parameters"], dim: b["parameters"][dim]}) != a["trial_hash"]
