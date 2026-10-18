"""Research snapshot resolution, content addressing and fixture/research contamination scan (spec 4.6, INV-DATA-001): offline."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quant_pipeline import collector as C
from quant_pipeline.cli import app
from quant_pipeline.config import Collector, load_config
from quant_pipeline.data_quality import validate_data
from quant_pipeline.snapshot import SnapshotResolutionError, hydrate_universe, resolve_snapshot, scan_contamination

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "fixtures" / "fixture_3asset"
FIX_CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
RENAME = {"KRX_A": "R00", "KRX_B": "R01", "NYSE_C": "R100"}
START, END = "2023-12-28", "2024-05-21"
SETTINGS = Collector(enabled=True, provider="local_files", cache_root="unused", requests_per_minute=600, max_concurrency=2, retries=0,
                     backoff_seconds=0.0, fail_on_partial=True)


def research_cfg():
    return load_config(ROOT / "configs" / "research_8_12asset.yaml")


def source_dir(tmp_path: Path, rename: dict[str, str]) -> Path:
    """Fixture-shaped canonical tables with instrument ids renamed: stands in for provider output (synthetic, test only)."""
    src = tmp_path / "source"
    src.mkdir()
    for stem in C.SCHEMAS:
        df = pd.read_parquet(FIXTURE / f"{stem}.parquet")
        for col in ("instrument_id",):
            if col in df.columns:
                df[col] = df[col].replace(rename)
        df.to_parquet(src / f"{stem}.parquet", index=False)
    return src


def universe_file(tmp_path: Path, rename: dict[str, str]) -> Path:
    assets = []
    for a in FIX_CFG.universe.assets:
        d = a.model_dump()
        d["asset_id"] = rename.get(a.asset_id, a.asset_id)
        assets.append(d)
    for i in range(2, 5):  # a complete 8-asset universe (spec: 8-12); only the first three have data in this tiny snapshot
        assets.append({**assets[0], "asset_id": f"R{i:02d}"})
    for i in range(101, 106):
        assets.append({**assets[2], "asset_id": f"R{i}"})
    path = tmp_path / "universe_source.yaml"
    path.write_text(yaml.safe_dump({"assets": assets}, sort_keys=False), encoding="utf-8")
    return path


def build_repo(tmp_path: Path, rename: dict[str, str] | None = RENAME, *, link: bool = True) -> tuple[Path, str]:
    """A repository root holding a collected research snapshot; ``current`` is a symlink to its content-addressed directory."""
    rename = RENAME if rename is None else rename
    assets = [a.model_copy(update={"asset_id": rename.get(a.asset_id, a.asset_id)}) for a in FIX_CFG.universe.assets]
    out = tmp_path / "data" / "snapshots" / "research" / "<snapshot_id>"
    report = C.collect(assets, fx_pair=FIX_CFG.fx.pair_id, rates_symbol="krw_rfr", collector=SETTINGS,
                       provider=C.LocalFilesProvider(source_dir(tmp_path, rename), "synthetic test data; no third-party rights"),
                       start=START, end=END, output=out, cache_root=tmp_path / "cache", jitter_seed=1731,
                       now=lambda: "2026-01-01T00:00:00Z", universe_source=universe_file(tmp_path, rename))
    assert report.status == "success", report.issues
    sid = report.snapshot_id.removeprefix("sha256:")
    (tmp_path / "data" / "fixtures" / "fixture_3asset").mkdir(parents=True)  # the scan compares against the repository's fixture manifest
    shutil.copy(FIXTURE / "manifest.json", tmp_path / "data" / "fixtures" / "fixture_3asset" / "manifest.json")
    if link:
        (tmp_path / "data" / "snapshots" / "research" / "current").symlink_to(sid, target_is_directory=True)
    return tmp_path, sid


def snap_dir(root: Path, sid: str) -> Path:
    return root / "data" / "snapshots" / "research" / sid


def test_the_collector_snapshot_carries_the_universe_and_default_cash_holdings_under_the_manifest_hash(tmp_path):
    root, sid = build_repo(tmp_path)
    d = snap_dir(root, sid)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert {"universe.yaml", "initial_holdings.csv"} <= set(manifest["files"])
    assert C.verify_snapshot(d) == f"sha256:{sid}"
    h = pd.read_csv(d / "initial_holdings.csv", dtype=str, keep_default_na=False)
    assert list(h.asset_id) == ["CASH_KRW", "R00", "R01", "R100"] and h.cash_amount.iloc[0] == "1000000000.00"
    assert set(h.as_of_utc) == {f"{START}T00:00:00Z"} and set(h.quantity) == {"0"}


def test_the_pointer_resolves_to_the_content_addressed_directory(tmp_path):
    root, sid = build_repo(tmp_path)
    snap = resolve_snapshot(research_cfg(), root)
    assert snap.snapshot_id == f"sha256:{sid}" and snap.directory == snap_dir(root, sid).resolve() and snap.directory.name == sid
    assert set(snap.files) >= {"manifest", "universe", "prices_raw", "initial_holdings"} and all(p.parent == snap.directory for p in snap.files.values())


def test_a_plain_directory_named_current_is_not_a_content_addressed_snapshot(tmp_path):
    root, sid = build_repo(tmp_path, link=False)
    shutil.copytree(snap_dir(root, sid), root / "data" / "snapshots" / "research" / "current")
    with pytest.raises(SnapshotResolutionError, match="not the content-addressed"):
        resolve_snapshot(research_cfg(), root)


def test_the_run_identity_follows_the_resolved_path_so_a_moved_pointer_is_a_new_lineage(tmp_path):
    root, sid = build_repo(tmp_path)
    other = tmp_path / "second"
    other.mkdir()
    root2, sid2 = build_repo(other, {"KRX_A": "R10", "KRX_B": "R11", "NYSE_C": "R101"})
    assert sid != sid2
    link = root / "data" / "snapshots" / "research" / "current"
    shutil.copytree(snap_dir(root2, sid2), snap_dir(root, sid2))
    link.unlink()
    link.symlink_to(sid2, target_is_directory=True)
    assert resolve_snapshot(research_cfg(), root).snapshot_id == f"sha256:{sid2}"


@pytest.mark.parametrize("what", ["tamper_data", "tamper_manifest", "delete_file", "wrong_directory_name"])
def test_a_snapshot_that_does_not_verify_is_refused(tmp_path, what):
    root, sid = build_repo(tmp_path)
    d = snap_dir(root, sid)
    if what == "tamper_data":
        (d / "prices_raw.parquet").write_bytes((d / "prices_raw.parquet").read_bytes() + b"x")
    elif what == "tamper_manifest":
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        m["provider"]["license"] = "changed"
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    elif what == "delete_file":
        (d / "rates_krw_rfr.parquet").unlink()
    else:
        link = root / "data" / "snapshots" / "research" / "current"
        link.unlink()
        shutil.copytree(d, d.parent / ("0" * 64))
        link.symlink_to("0" * 64, target_is_directory=True)
    with pytest.raises(SnapshotResolutionError):
        resolve_snapshot(research_cfg(), root)


def test_files_read_by_the_profile_must_be_covered_by_the_manifest_hash(tmp_path):
    root, sid = build_repo(tmp_path)
    d = snap_dir(root, sid)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    del m["files"]["initial_holdings.csv"]
    m.pop("snapshot_id")
    from quant_pipeline.hashing import hash_json_obj
    new = hash_json_obj(m)
    (d / "manifest.json").write_text(json.dumps({**m, "snapshot_id": new}), encoding="utf-8")
    nd = d.parent / new.removeprefix("sha256:")
    shutil.copytree(d, nd)
    link = d.parent / "current"
    link.unlink()
    link.symlink_to(nd.name, target_is_directory=True)
    with pytest.raises(SnapshotResolutionError, match="not covered by the manifest hash"):
        resolve_snapshot(research_cfg(), root)


def test_a_missing_pointer_and_a_fixture_profile_are_reported(tmp_path):
    with pytest.raises(SnapshotResolutionError, match="does not exist"):
        resolve_snapshot(research_cfg(), tmp_path)
    with pytest.raises(SnapshotResolutionError, match="only snapshot profiles"):
        resolve_snapshot(FIX_CFG, ROOT)


def test_a_clean_synthetic_snapshot_passes_the_contamination_scan(tmp_path):
    root, _ = build_repo(tmp_path)
    scan = scan_contamination(research_cfg(), root, resolve_snapshot(research_cfg(), root))
    assert all(ok for _, ok, _ in scan), scan
    assert {n for n, _, _ in scan} >= {"contamination_path_roots", "contamination_no_shared_file_content", "contamination_no_fixture_instruments"}


def test_fixture_instrument_ids_in_a_research_snapshot_are_contamination(tmp_path):
    root, _ = build_repo(tmp_path, rename={"KRX_A": "KRX_A", "KRX_B": "R01", "NYSE_C": "R100"})  # one fixture id stays
    failed = {n for n, ok, _ in scan_contamination(research_cfg(), root, resolve_snapshot(research_cfg(), root)) if not ok}
    assert "contamination_no_fixture_instruments" in failed


def test_a_fixture_file_copied_into_a_research_snapshot_is_detected(tmp_path):
    root, _ = build_repo(tmp_path)
    snap = resolve_snapshot(research_cfg(), root)
    fx = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    shared = dict(snap.manifest["files"])
    shared["prices_raw.parquet"] = {**shared["prices_raw.parquet"], "raw_sha256": next(iter(fx["files"].values()))["raw_sha256"]}
    scan = scan_contamination(research_cfg(), root, type(snap)(snap.snapshot_id, snap.directory, snap.files, {**snap.manifest, "files": shared}))
    assert "contamination_no_shared_file_content" in {n for n, ok, _ in scan if not ok}


def test_fixture_generation_keys_and_regime_tables_in_a_manifest_are_contamination(tmp_path):
    root, _ = build_repo(tmp_path)
    snap = resolve_snapshot(research_cfg(), root)
    dirty = type(snap)(snap.snapshot_id, snap.directory, snap.files,
                       {**snap.manifest, "scenario": "fixture-v5", "files": {**snap.manifest["files"], "fixed_regime_table.csv": {}}})
    failed = {n for n, ok, _ in scan_contamination(research_cfg(), root, dirty) if not ok}
    assert {"contamination_no_fixture_provenance", "contamination_no_fixture_regime_table"} <= failed


def test_data_outside_the_universe_is_contamination(tmp_path):
    root, _ = build_repo(tmp_path)
    snap = resolve_snapshot(research_cfg(), root)
    u = yaml.safe_load(snap.files["universe"].read_text(encoding="utf-8"))
    u["assets"] = u["assets"][:2]
    patched = tmp_path / "u.yaml"
    patched.write_text(yaml.safe_dump(u), encoding="utf-8")
    files = {**snap.files, "universe": patched}
    failed = {n for n, ok, _ in scan_contamination(research_cfg(), root, type(snap)(snap.snapshot_id, snap.directory, files, snap.manifest)) if not ok}
    assert "contamination_instruments_in_universe" in failed


def test_the_verified_universe_records_hydrate_the_config(tmp_path):
    root, _ = build_repo(tmp_path)
    cfg = research_cfg()
    assert cfg.universe.assets is None
    hydrated = hydrate_universe(cfg, resolve_snapshot(cfg, root))
    assert [a.asset_id for a in hydrated.universe.assets][:2] == ["R00", "R01"] and len(hydrated.universe.assets) == 11 and cfg.universe.assets is None  # the original is untouched


def test_data_validate_refuses_an_unresolvable_snapshot_before_reading_any_table(tmp_path):
    root, sid = build_repo(tmp_path)
    d = snap_dir(root, sid)
    (d / "prices_raw.parquet").write_bytes(b"corrupt")
    failed = {r["check"] for r in validate_data(research_cfg(), root) if r["status"] != "pass"}
    assert "snapshot_resolved" in failed and "raw_unique_keys" not in failed  # nothing after the resolution gate ran


def test_data_validate_reaches_the_table_checks_for_a_verified_snapshot(tmp_path):
    root, _ = build_repo(tmp_path)
    names = {r["check"] for r in validate_data(research_cfg(), root)}
    assert {"snapshot_resolved", "contamination_path_roots", "raw_unique_keys", "manifest_integrity"} <= names


def test_the_cli_exposes_the_resolution_failure_as_a_nonzero_exit(tmp_path):
    root, sid = build_repo(tmp_path)
    rates = snap_dir(root, sid) / "rates_krw_rfr.parquet"
    rates.write_bytes(rates.read_bytes() + b"x")
    r = CliRunner().invoke(app, ["data", "validate", "--config", str(ROOT / "configs" / "research_8_12asset.yaml"), "--repo-root", str(root)])
    assert r.exit_code != 0 and "snapshot_resolved" in r.output


def test_a_snapshot_is_labelled_synthetic_only_when_the_collector_is_told_so(tmp_path):
    root, _ = build_repo(tmp_path)
    plain = resolve_snapshot(research_cfg(), root)
    assert plain.synthetic is False and "synthetic" not in plain.manifest  # existing snapshot ids are not affected

    sys_path = str(ROOT / "tests")
    import sys

    sys.path.insert(0, sys_path)
    from support.synthetic_research import build_research_repo

    other = tmp_path / "syn"
    other.mkdir()
    root2, _ = build_research_repo(other, start="2016-01-04", end="2016-03-31")
    assert resolve_snapshot(research_cfg(), root2).synthetic is True
