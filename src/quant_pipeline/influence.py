"""Paired-trial dependency influence matrix (spec 3.4, check ``INV-DEP-001``).

For every trial dimension, two trials that differ in that dimension alone are fingerprinted per artifact type. The matrix records
whether the cache key moved, which must agree with the declared ``relevant_parameter_subset`` of each artifact.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import Config
from .decision import TrialParams
from .dependency import build_fingerprint, cache_key, default_numerical_options
from .engine import PathEngine

# the only dimensions each artifact may depend on (spec 3.4); the core fit depends on none
DECLARED: dict[str, tuple[str, ...]] = {
    "core_fit": (),
    "objective_calibration": ("overlay_gamma", "entry_z", "horizon_weights", "gate_name"),
    "optimizer_decision": ("overlay_gamma", "entry_z", "horizon_weights", "gate_name", "lambda_track", "lambda_risk", "lambda_cost"),
}
DIMENSIONS = tuple(DECLARED["optimizer_decision"])


def _subset(artifact: str, t: TrialParams) -> dict[str, Any]:
    names = {"gate_name": "gate"}  # the production key spells the gate dimension "gate"
    return {names.get(d, d): (list(getattr(t, d)) if d == "horizon_weights" else getattr(t, d)) for d in DECLARED[artifact]}


def _key(artifact: str, t: TrialParams, hashes: dict[str, str]) -> str:
    sub = _subset(artifact, t)
    if artifact == "objective_calibration" and sub != PathEngine._target_subset(t):
        raise AssertionError("the declared calibration subset differs from the production key subset")
    return cache_key(build_fingerprint(
        artifact_type=artifact, schema_version="5.3.2", algorithm_component_version="influence-1", spec_hash=hashes["spec_hash"],
        code_hash_=hashes["code_hash"], lock_hash=None, data_manifest_hash=hashes["data_manifest_hash"],
        base_profile_hash=hashes["base_profile_hash"], asset_order=("A",), information_cutoff_utc="t", fit_start_utc="t",
        fit_end_utc="t", observation_set_hash="o", upstream_content_hashes={}, relevant_parameter_subset=sub, seed=None,
        solver_versions={}, numerical_options=default_numerical_options(),
        trial_influence={"influenced": bool(sub), "parameters": sorted(sub)}))


def _alternative(base: TrialParams, dim: str) -> TrialParams:
    hw = base.horizon_weights
    other = {"overlay_gamma": base.overlay_gamma + 0.5, "entry_z": base.entry_z + 0.5,
             "horizon_weights": (*hw[:-1], hw[-1] + 0.1), "gate_name": base.gate_name + "_alt",
             "lambda_track": base.lambda_track + 1.0, "lambda_risk": base.lambda_risk + 1.0, "lambda_cost": base.lambda_cost + 1.0}
    return replace(base, **{dim: other[dim]})


def influence_matrix(cfg: Config, hashes: dict[str, str]) -> dict[str, Any]:
    base = TrialParams.first(cfg)
    rows, violations = [], []
    for dim in DIMENSIONS:
        alt = _alternative(base, dim)
        row: dict[str, Any] = {"dimension": dim}
        for artifact, declared in DECLARED.items():
            moved = _key(artifact, base, hashes) != _key(artifact, alt, hashes)
            row[artifact] = moved
            if moved != (dim in declared):
                violations.append({"dimension": dim, "artifact": artifact, "key_moved": moved, "declared": dim in declared})
        rows.append(row)
    return {"schema": "dependency-influence-1", "base_profile_hash": hashes["base_profile_hash"], "dimensions": list(DIMENSIONS),
            "artifacts": list(DECLARED), "rows": rows, "violations": violations}
