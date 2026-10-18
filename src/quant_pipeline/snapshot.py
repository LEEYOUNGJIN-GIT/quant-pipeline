"""Research snapshot resolution and fixture/research contamination scan (spec 4.6, 5, INV-DATA-001).

``data/snapshots/research/current`` is only a convenience pointer. Before any computation it must resolve to an immutable directory
``data/snapshots/research/<snapshot_id>/`` whose name equals the verified canonical hash of its manifest; the resolved path and id, not
``current``, identify the run. Nothing here reads the network or modifies a snapshot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .collector import SnapshotError, verify_snapshot
from .config import Config

REQUIRED_KIND = "research"
FIXTURE_ASSET_IDS = frozenset({"KRX_A", "KRX_B", "NYSE_C"})
FIXTURE_MANIFEST_KEYS = ("scenario", "generation_lock", "calendar_session_digest", "generator_sha256", "seed")


class SnapshotResolutionError(ValueError):
    """The configured snapshot is missing, not content-addressed, incomplete, or not a research snapshot."""


@dataclass(frozen=True)
class ResolvedSnapshot:
    snapshot_id: str  # sha256:<hex>, the manifest's canonical hash
    directory: Path  # resolved, content-addressed
    files: dict[str, Path]  # configured role -> resolved path
    manifest: dict

    @property
    def synthetic(self) -> bool:
        return self.manifest.get("synthetic") is True


def configured_files(cfg: Config) -> dict[str, str]:
    d = cfg.data
    roles = {"manifest": d.manifest, "universe": d.universe_manifest, "prices_raw": d.prices_raw, "prices_total_return": d.prices_total_return,
             "fx": d.fx, "rates": d.rates, "corporate_actions": d.corporate_actions, "initial_holdings": d.initial_holdings}
    return {k: v for k, v in roles.items() if v}


def resolve_snapshot(cfg: Config, root: Path) -> ResolvedSnapshot:
    """Resolve the configured pointer to its immutable directory and verify every referenced file; raise ``SnapshotResolutionError``."""
    if cfg.data.mode != "snapshot":
        raise SnapshotResolutionError(f"data.mode={cfg.data.mode}: only snapshot profiles resolve a research snapshot")
    paths = configured_files(cfg)
    pointers = {Path(v).parent for v in paths.values()}
    if len(pointers) != 1:
        raise SnapshotResolutionError(f"the profile reads from several directories: {sorted(str(p) for p in pointers)}")
    pointer = root / next(iter(pointers))
    if not pointer.exists():
        raise SnapshotResolutionError(f"{pointer.relative_to(root)} does not exist")
    directory = pointer.resolve()
    try:
        snapshot_id = verify_snapshot(directory)
    except SnapshotError as e:
        raise SnapshotResolutionError(str(e)) from e
    if directory.name != snapshot_id.removeprefix("sha256:"):
        raise SnapshotResolutionError(
            f"{pointer.relative_to(root)} resolves to '{directory.name}', which is not the content-addressed '{snapshot_id.removeprefix('sha256:')}' directory "
            "(spec 4.6: the pointer must resolve to data/snapshots/research/<snapshot_id>/)")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("snapshot_kind") != REQUIRED_KIND:
        raise SnapshotResolutionError(f"manifest snapshot_kind is {manifest.get('snapshot_kind')!r}, expected {REQUIRED_KIND!r}")
    names = {role: Path(v).name for role, v in paths.items() if role != "manifest"}
    uncovered = sorted(n for n in names.values() if n not in manifest["files"])
    if uncovered:
        raise SnapshotResolutionError(f"files read by the profile are not covered by the manifest hash: {uncovered}")
    return ResolvedSnapshot(snapshot_id, directory, {role: directory / Path(v).name for role, v in paths.items()}, manifest)


def hydrate_universe(cfg: Config, snap: ResolvedSnapshot) -> Config:
    """A copy of ``cfg`` whose universe records come from the verified snapshot's ``universe.yaml`` (``universe.source: manifest``)."""
    from .config import Asset, load_yaml_strict

    records = load_yaml_strict(snap.files["universe"].read_text(encoding="utf-8"))["assets"]
    assets = [Asset.model_validate(r) for r in records]
    return cfg.model_copy(update={"universe": cfg.universe.model_copy(update={"assets": assets})})


def load_research_context(cfg: Config, root: Path) -> tuple[Config, ResolvedSnapshot]:
    """Resolve, contamination-scan and hydrate: the only way a research profile gets its universe. Raises ``SnapshotResolutionError``."""
    snap = resolve_snapshot(cfg, root)
    failed = [f"{name}: {detail}" for name, ok, detail in scan_contamination(cfg, root, snap) if not ok]
    if failed:
        raise SnapshotResolutionError("contamination scan failed: " + "; ".join(failed))
    return hydrate_universe(cfg, snap), snap


MAX_UNOBSERVED_SHARE = 0.005  # of the common XKRX/XNYS sessions: more missing provider data than this is a data problem, not a gap
MAX_UNOBSERVED_RUN = 3  # consecutive common sessions


def unobserved_common_sessions(cfg: Config, snap: ResolvedSnapshot, common_sessions: pd.DatetimeIndex) -> frozenset[str]:
    """Common XKRX/XNYS sessions on which at least one risky asset has no accepted total-return observation.

    Spec 5.1 rules 5-7: such a row has a null return for that asset and is excluded from the complete-case rows; nothing is filled, backfilled
    or spread over other rows, and the asset's next return runs from its preceding observed level. Derived from the verified snapshot only."""
    tr = pd.read_parquet(snap.files["prices_total_return"], columns=["instrument_id", "exchange_session"])
    have = tr.groupby("instrument_id").exchange_session.agg(set)
    want = {s.strftime("%Y-%m-%d") for s in common_sessions}
    out: set[str] = set()
    for a in cfg.universe.assets or []:
        out |= want - have.get(a.asset_id, set())
    return frozenset(out)


def research_unobserved(cfg: Config, snap: ResolvedSnapshot, cals) -> frozenset[str]:
    """``unobserved_common_sessions`` over the snapshot's own collected window; what the research row map excludes."""
    from .calendars import window_from_manifest

    start, end = window_from_manifest(snap.manifest)
    both = cals.sessions("XNYS", start, end).intersection(cals.sessions("XKRX", start, end))
    return unobserved_common_sessions(cfg, snap, both)


def complete_case_coverage(cfg: Config, snap: ResolvedSnapshot, common_sessions: pd.DatetimeIndex) -> list[tuple[str, bool, str]]:
    """The common sessions without a full set of observations are excluded from the complete-case rows (listed, never filled); too many, or too
    many in a row, is a data problem and fails."""
    missing = sorted(unobserved_common_sessions(cfg, snap, common_sessions))
    order = [s.strftime("%Y-%m-%d") for s in common_sessions]
    flags = [d in set(missing) for d in order]
    run = best = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    share = len(missing) / max(len(order), 1)
    ok = share <= MAX_UNOBSERVED_SHARE and best <= MAX_UNOBSERVED_RUN
    if not missing:
        return [("complete_case_coverage", True, "every asset observes every common session")]
    detail = (f"{len(missing)} of {len(order)} common sessions ({share:.4%}) lack an observation for some asset and are excluded from the complete-case rows "
              f"(longest run {best}; limits {MAX_UNOBSERVED_SHARE:.2%} and {MAX_UNOBSERVED_RUN}): {' '.join(missing[:12])}")
    return [("complete_case_coverage", ok, detail)]


def scan_contamination(cfg: Config, root: Path, snap: ResolvedSnapshot) -> list[tuple[str, bool, str]]:
    """Fixture/research isolation (spec 12.6 'fixture/research contamination'): ``(check, ok, detail)`` rows; every one must pass."""
    out: list[tuple[str, bool, str]] = []
    rel = snap.directory.relative_to(root.resolve()).parts if snap.directory.is_relative_to(root.resolve()) else snap.directory.parts
    out.append(("contamination_path_roots", "snapshots" in rel and "fixtures" not in rel, "/".join(rel)))

    fixture_manifest = root / "data" / "fixtures" / "fixture_3asset" / "manifest.json"
    if fixture_manifest.exists():
        fm = json.loads(fixture_manifest.read_text(encoding="utf-8"))
        fx_raw = {m["raw_sha256"] for m in fm.get("files", {}).values()}
        shared = sorted(n for n, m in snap.manifest["files"].items() if m.get("raw_sha256") in fx_raw)
        out.append(("contamination_no_shared_file_content", not shared, f"files identical to a fixture file: {shared}" if shared else "no file content is shared"))
        out.append(("contamination_distinct_manifests", fm.get("snapshot_id") != snap.snapshot_id, "snapshot id differs from the fixture manifest"))
    leaked_keys = sorted(k for k in FIXTURE_MANIFEST_KEYS if k in snap.manifest)
    out.append(("contamination_no_fixture_provenance", not leaked_keys, f"fixture-only manifest keys present: {leaked_keys}" if leaked_keys else "no fixture generation keys"))
    out.append(("contamination_no_fixture_regime_table", not any("regime" in n for n in snap.manifest["files"]),
                "the research snapshot carries no fixed regime table"))

    ids: set[str] = set()
    for role in ("prices_raw", "prices_total_return", "corporate_actions"):
        ids |= set(pd.read_parquet(snap.files[role], columns=["instrument_id"]).instrument_id)
    leaked = sorted(ids & FIXTURE_ASSET_IDS)
    out.append(("contamination_no_fixture_instruments", not leaked, f"fixture instrument ids in data: {leaked}" if leaked else "no fixture instrument ids"))
    from .config import load_yaml_strict

    known = {r["asset_id"] for r in load_yaml_strict(snap.files["universe"].read_text(encoding="utf-8"))["assets"]}
    unknown = sorted(ids - known)
    out.append(("contamination_instruments_in_universe", not unknown, f"data instruments missing from the universe: {unknown}" if unknown else "every data instrument is a universe asset"))
    return out
