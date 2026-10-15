from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_pipeline.config import Config, load_config, load_yaml_strict

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "configs" / "fixture_3asset.yaml"
RESEARCH = ROOT / "configs" / "research_8_12asset.yaml"


def raw(path: Path) -> dict:
    return load_yaml_strict(path.read_text(encoding="utf-8"))


def test_both_profiles_validate_under_one_schema():
    fx, rs = load_config(FIXTURE), load_config(RESEARCH)
    assert fx.profile.profile_type == "fixture" and rs.profile.profile_type == "research"
    assert list(raw(FIXTURE)) == list(raw(RESEARCH))  # identical root keys, same order


def test_valuation_and_freshness_separation():
    fx, rs = load_config(FIXTURE), load_config(RESEARCH)
    assert (fx.valuation.max_krx_stale_sessions, rs.valuation.max_krx_stale_sessions) == (2, 6)
    for c in (fx, rs):
        assert c.availability.max_price_stale_sessions == 2 and c.panel.max_stale_sessions == 2


def test_fixture_bootstrap_contract():
    b = load_config(FIXTURE).bootstrap
    assert (b.replicates, b.expected_oos_rows, b.expected_index_shape) == (200, 22, [200, 22])


def test_research_has_four_gate_tables():
    names = {g.name for g in load_config(RESEARCH).regime.agent_gate_candidates}
    assert {"balanced_2state", "balanced_3state", "defensive_tilt_2state", "defensive_tilt_3state"} <= names


def test_unknown_key_rejected():
    d = raw(FIXTURE)
    d["regime"]["decision_mode"] = "x"  # forbidden alias
    with pytest.raises(ValidationError):
        Config.model_validate(d)


@pytest.mark.parametrize("path,value", [
    (("evaluation", "purge", "mode"), "auto"),
    (("agents", "trend", "horizon_weights"), [[1.0]]),
    (("objective", "lambda_track"), 1.0),
])
def test_invalid_aliases_rejected(path, value):
    d = copy.deepcopy(raw(FIXTURE))
    node = d
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_duplicate_yaml_key_rejected():
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_yaml_strict("a: 1\na: 2\n")


def test_non_finite_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        load_yaml_strict("a: .inf\n")


def test_solver_attempts_must_be_two():
    d = copy.deepcopy(raw(FIXTURE))
    d["compute_budget"]["max_solver_attempts_per_problem"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_research_valuation_6_not_allowed_in_availability():
    d = copy.deepcopy(raw(RESEARCH))
    d["availability"]["max_price_stale_sessions"] = 6
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_hmm_mode_forbids_fixed_table():
    d = copy.deepcopy(raw(RESEARCH))
    d["regime"]["fixed_table"] = {"path": "x.csv"}
    with pytest.raises(ValidationError):
        Config.model_validate(d)
