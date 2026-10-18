"""Checkpoint store (spec 12.2, 12.3): keyed by (run, fold, trial, stage, refit), written atomically, validated on read.

A checkpoint is valid only if its completion marker exists, ``checkpoint.json`` parses with the supported schema, every named
artifact exists with the recorded hash, and the lineage (spec/config/data/lock/code hashes) equals the caller's. Prior artifacts
are never overwritten.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hashing import hash_json_obj, sha256_bytes

SCHEMA = "checkpoint-1"
MARKER = "COMPLETE"
STAGES = (
    "00_capability_probe", "01_config_and_manifest", "02_ingestion_and_quality", "03_canonical_panel", "04_features_and_pca",
    "05_covariance_and_asynchrony", "06_hmm_filtered_state", "07_agent_outputs", "08_aggregation_and_optimization",
    "09_rebalance_plan", "10_event_time_execution_and_accounting", "11_oos_evaluation", "12_bootstrap_and_reports", "13_verification",
)


class LineageMismatch(RuntimeError):
    """A material mismatch (spec/config/data/lock/code): resuming would mix lineages, so a new lineage must be started."""


class CheckpointExists(FileExistsError):
    """Prior artifacts are never overwritten."""


@dataclass(frozen=True)
class CheckpointKey:
    run_id: str
    fold_id: str
    trial_hash: str
    stage: str
    refit_id: str

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        for part in (self.run_id, self.fold_id, self.trial_hash, self.refit_id):
            if not part or any(c in part for c in "/\\"):
                raise ValueError(f"invalid checkpoint key component {part!r}")

    def relpath(self) -> Path:
        # ':' is replaced so the identifier is a legal directory name on every platform
        safe = lambda s: s.replace(":", "_")
        return (Path(self.run_id) / "checkpoints" / f"fold_id={safe(self.fold_id)}" / f"trial_hash={safe(self.trial_hash)}"
                / f"stage={self.stage}" / f"refit_id={safe(self.refit_id)}")

    def ident(self) -> str:
        return "|".join(asdict(self).values())


class CheckpointStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: CheckpointKey) -> Path:
        """Spec layout under the store root. On Windows every path gets the extended-length prefix (MAX_PATH is 260)."""
        p = (self.root / key.relpath()).resolve()
        prefix = "\\\\?\\"  # the Windows extended-length path prefix: two backslashes, a question mark, one backslash
        # always on Windows: a directory under the limit can still hold files whose full path is over it
        if os.name == "nt" and not str(p).startswith(prefix):
            p = Path(prefix + str(p))
        return p

    # ---------------------------------------------------------------------------------------------------- write
    def write(self, key: CheckpointKey, artifacts: dict[str, bytes], *, lineage: dict[str, str], parents: list[CheckpointKey] | None = None,
              meta: dict[str, Any] | None = None) -> Path:
        final = self.path(key)
        if final.exists():
            raise CheckpointExists(f"checkpoint already exists: {key.ident()}")
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            manifest = {}
            for name, data in sorted(artifacts.items()):
                if "/" in name or "\\" in name or name in (MARKER, "checkpoint.json"):
                    raise ValueError(f"invalid artifact name {name!r}")
                fp = tmp / name
                with fp.open("wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                manifest[name] = sha256_bytes(data)
            body = {"schema": SCHEMA, "key": asdict(key), "lineage": lineage, "parents": [asdict(p) for p in parents or []],
                    "artifacts": manifest, "meta": meta or {}}
            body["content_hash"] = hash_json_obj({k: v for k, v in body.items()})
            (tmp / "checkpoint.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
            marker = tmp / MARKER  # written last: its presence is the completion proof
            marker.write_text(body["content_hash"], encoding="utf-8")
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return final

    # ---------------------------------------------------------------------------------------------------- read / validate
    def validate(self, key: CheckpointKey, lineage: dict[str, str]) -> tuple[bool, str]:
        d = self.path(key)
        if not d.is_dir():
            return False, "missing"
        if not (d / MARKER).exists():
            return False, "no_completion_marker"
        try:
            body = json.loads((d / "checkpoint.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, "unreadable_checkpoint_json"
        if body.get("schema") != SCHEMA:
            return False, "schema_mismatch"
        recorded = body.pop("content_hash", None)
        if hash_json_obj(body) != recorded or (d / MARKER).read_text(encoding="utf-8") != recorded:
            return False, "content_hash_mismatch"
        for name, h in body["artifacts"].items():
            f = d / name
            if not f.is_file():
                return False, f"missing_artifact:{name}"
            if sha256_bytes(f.read_bytes()) != h:
                return False, f"artifact_hash_mismatch:{name}"
        if body["lineage"] != lineage:
            diff = sorted(k for k in set(body["lineage"]) | set(lineage) if body["lineage"].get(k) != lineage.get(k))
            return False, f"lineage_mismatch:{','.join(diff)}"
        return True, "ok"

    def read(self, key: CheckpointKey, name: str, lineage: dict[str, str]) -> bytes:
        ok, why = self.validate(key, lineage)
        if not ok:
            raise LineageMismatch(why) if why.startswith("lineage_mismatch") else ValueError(f"invalid checkpoint {key.ident()}: {why}")
        return (self.path(key) / name).read_bytes()

    def meta(self, key: CheckpointKey) -> dict[str, Any]:
        return json.loads((self.path(key) / "checkpoint.json").read_text(encoding="utf-8"))["meta"]

    def latest_valid(self, chain: list[CheckpointKey], lineage: dict[str, str]) -> int:
        """Number of leading checkpoints of a dependency chain that are valid (resume continues after the last one).

        Resume follows the exact dependency graph, not the newest timestamp: validity of an entry also requires the
        validity of every earlier entry in the chain. A lineage mismatch raises instead of silently starting over.
        """
        n = 0
        for key in chain:
            ok, why = self.validate(key, lineage)
            if not ok:
                if why.startswith("lineage_mismatch"):
                    raise LineageMismatch(why)
                break
            n += 1
        return n

    def discard_invalid(self, key: CheckpointKey, lineage: dict[str, str]) -> str | None:
        """Move an invalid (partial/corrupt) checkpoint aside so it can be recomputed; valid ones are never touched."""
        ok, why = self.validate(key, lineage)
        d = self.path(key)
        if ok or not d.exists():
            return None
        if why.startswith("lineage_mismatch"):
            raise LineageMismatch(why)
        quarantine = d.parent / f".invalid-{d.name}-{uuid.uuid4().hex[:6]}"
        os.replace(d, quarantine)
        return why
