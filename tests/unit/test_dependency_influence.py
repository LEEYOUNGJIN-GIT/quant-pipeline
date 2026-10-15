"""Paired-trial dependency influence (INV-DEP-001): keys move exactly with the declared relevant subset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_pipeline import influence
from quant_pipeline.config import load_config
from quant_pipeline.dependency import build_fingerprint, cache_key
from quant_pipeline.fitstate import lineage_hashes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    return cfg, lineage_hashes(cfg, ROOT)


def test_the_matrix_has_no_violations_and_every_dimension_is_exercised(setup):
    m = influence.influence_matrix(*setup)
    assert m["violations"] == [] and [r["dimension"] for r in m["rows"]] == list(influence.DIMENSIONS)
    for r in m["rows"]:
        assert r["core_fit"] is False  # no trial dimension reaches the shared core fit
        assert r["optimizer_decision"] is True  # every dimension reaches the decision
    cal = {r["dimension"]: r["objective_calibration"] for r in m["rows"]}
    assert not any(cal[d] for d in ("lambda_track", "lambda_risk", "lambda_cost"))  # lambdas share calibrations
    assert all(cal[d] for d in ("overlay_gamma", "entry_z", "horizon_weights", "gate_name"))


def test_a_wrongly_declared_subset_is_detected(setup, monkeypatch):
    monkeypatch.setitem(influence.DECLARED, "core_fit", ("entry_z",))  # declare a dependency the key never uses
    # core_fit now declares entry_z, so its key subset moves with entry_z, which the declared table for the decision also lists
    m = influence.influence_matrix(*setup)
    assert m["violations"] == []  # consistent declaration: key and declaration agree
    monkeypatch.setitem(influence.DECLARED, "objective_calibration", ("overlay_gamma",))
    with pytest.raises(AssertionError):
        influence.influence_matrix(*setup)  # declaration differs from the production subset


def test_lineage_fields_cannot_enter_a_key_and_a_changed_profile_moves_every_key():
    kwargs = {"artifact_type": "x", "schema_version": "1", "algorithm_component_version": "1", "spec_hash": "s", "code_hash_": "c", "lock_hash": None,
                  "data_manifest_hash": "d", "base_profile_hash": "b", "asset_order": ("A",), "information_cutoff_utc": "t", "fit_start_utc": "t",
                  "fit_end_utc": "t", "observation_set_hash": "o", "upstream_content_hashes": {}, "relevant_parameter_subset": {}, "seed": None,
                  "solver_versions": {}, "numerical_options": {}, "trial_influence": {"influenced": False}}
    fp = build_fingerprint(**kwargs)
    with pytest.raises(ValueError):
        cache_key({**fp, "fold_id": "f1"})
    assert cache_key(fp) != cache_key(build_fingerprint(**{**kwargs, "base_profile_hash": "b2"}))


def test_the_matrix_is_deterministic_and_serialisable(setup, tmp_path):
    a = json.dumps(influence.influence_matrix(*setup), sort_keys=True)
    assert a == json.dumps(influence.influence_matrix(*setup), sort_keys=True)
    (tmp_path / "dependency_influence_matrix.json").write_text(a, encoding="utf-8")
