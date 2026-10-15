"""Typed dependency fingerprints and cache keys (spec 3.4).

Lineage fields (fold, refit, trial, run IDs and the whole post-overlay profile hash) are stored beside the artifact but do
not enter the key unless they change an actual input.
"""
from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
from typing import Any

from .hashing import hash_json_obj, hash_text

LINEAGE_FIELDS = ("fold_id", "inner_fold_id", "refit_id", "trial_id", "run_id", "profile_hash_post_overlay")


def accepted_lock_hash(root: Path) -> str:
    """SHA-256 of the accepted ``uv.lock``; an explicit unavailable marker when none is accepted (never a guess).

    A lock installed as a candidate (``QUANT_LOCK_CANDIDATE=1``) is not accepted until it passes verification on the target runtime.
    """
    lock = Path(root) / "uv.lock"
    if os.environ.get("QUANT_LOCK_CANDIDATE") == "1" or not lock.exists():
        return "unavailable:no_accepted_lock"
    return "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()


def code_hash(package_dir: Path) -> str:
    files = sorted(package_dir.glob("*.py"))
    return hash_json_obj({f.name: hash_text(f.read_bytes()) for f in files})


def build_fingerprint(*, artifact_type: str, schema_version: str, algorithm_component_version: str, spec_hash: str,
                      code_hash_: str, lock_hash: str | None, data_manifest_hash: str, base_profile_hash: str,
                      asset_order: tuple[str, ...], information_cutoff_utc: str, fit_start_utc: str, fit_end_utc: str,
                      observation_set_hash: str, upstream_content_hashes: dict[str, str],
                      relevant_parameter_subset: dict[str, Any], seed: dict[str, Any] | None,
                      solver_versions: dict[str, str], numerical_options: dict[str, Any],
                      trial_influence: dict[str, Any]) -> dict[str, Any]:
    """``lock_hash`` None means no accepted lock exists: stated explicitly, never invented."""
    return {
        "artifact_type": artifact_type, "schema_version": schema_version,
        "algorithm_component_version": algorithm_component_version, "spec_hash": spec_hash, "code_hash": code_hash_,
        "lock_hash": lock_hash if lock_hash else "unavailable:no_accepted_lock", "data_manifest_hash": data_manifest_hash,
        "base_profile_hash": base_profile_hash, "asset_order": list(asset_order),
        "information_cutoff_utc": information_cutoff_utc, "fit_start_utc": fit_start_utc, "fit_end_utc": fit_end_utc,
        "observation_set_hash": observation_set_hash, "upstream_content_hashes": dict(sorted(upstream_content_hashes.items())),
        "relevant_parameter_subset": relevant_parameter_subset, "relevant_seed_stream_and_value": seed,
        "calendar_and_solver_versions": solver_versions, "numerical_options": numerical_options,
        "trial_influence": trial_influence,
    }


def cache_key(fingerprint: dict[str, Any]) -> str:
    leaked = [f for f in LINEAGE_FIELDS if f in fingerprint]
    if leaked:
        raise ValueError(f"lineage fields must not enter the cache key: {leaked}")
    return hash_json_obj(fingerprint)


def default_numerical_options() -> dict[str, Any]:
    return {"numeric_threads": 1, "python": platform.python_version()}
