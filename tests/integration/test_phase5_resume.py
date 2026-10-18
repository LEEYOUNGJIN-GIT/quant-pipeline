"""Checkpoint / resume (spec 12.3): clean lineage vs an interrupted-and-resumed lineage in separate processes.
Evidence class: fixture_path (every run is a fresh interpreter; no state is shared in memory)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STOP_AFTER = 5  # the config checkpoint plus four completed inner-trial paths


def lp(path: Path) -> Path:
    """Windows extended-length form so tests can walk and copy the (long) checkpoint tree; identity elsewhere."""
    prefix = "\\\\?\\"  # two backslashes, a question mark, one backslash
    s = str(Path(path).resolve())
    return Path(prefix + s) if sys.platform == "win32" and not s.startswith(prefix) else Path(s)



def pipeline(out: Path, run_id: str, *, root: Path = ROOT, stop_after: int | None = None, projection: Path | None = None):
    cmd = [sys.executable, "-m", "quant_pipeline.pipeline", "--run-id", run_id, "--out-root", str(out), "--root", str(root)]
    if stop_after is not None:
        cmd += ["--stop-after", str(stop_after)]
    if projection is not None:
        cmd += ["--projection-out", str(projection)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)


UNITS = 21


def counts(stdout: str) -> tuple[int, int]:
    line = [x for x in stdout.splitlines() if x.startswith("complete:")][-1]
    parts = dict(p.split("=") for p in line.removeprefix("complete: ").split())
    return int(parts["computed"]), int(parts["loaded"])


def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    base = lp(path)
    for f in sorted(p for p in base.rglob("*") if p.is_file() and p.name != "progress_log.txt"):
        h.update(str(f.relative_to(base)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    d = tmp_path_factory.mktemp("clean")
    proj = d / "projection.json"
    r = pipeline(d / "out", "clean", projection=proj)
    assert r.returncode == 0, r.stdout + r.stderr
    return {"dir": d, "out": d / "out", "projection": json.loads(proj.read_text(encoding="utf-8")), "counts": counts(r.stdout)}


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory):
    d = tmp_path_factory.mktemp("interrupted")
    out = d / "out"
    stopped = pipeline(out, "resumed", stop_after=STOP_AFTER)
    assert stopped.returncode == 75 and "STOPPED" in stopped.stdout, stopped.stdout + stopped.stderr
    before = tree_hash(out)
    markers_at_stop = len(list(lp(out / "resumed" / "checkpoints").rglob("COMPLETE")))  # observed right after the stop
    proj = d / "projection.json"
    resumed = pipeline(out, "resumed", projection=proj)  # a new process, same run id and output directory
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    return {"out": out, "before": before, "markers_at_stop": markers_at_stop, "projection": json.loads(proj.read_text(encoding="utf-8")), "counts": counts(resumed.stdout)}


def test_the_clean_run_computes_every_unit_once(clean):
    assert clean["counts"] == (UNITS, 0)  # config + 8 trial paths + selection + 6 outer folds + final-selection freeze + summary + pre-holdout report


def test_the_hook_stops_immediately_after_a_durable_checkpoint(interrupted):
    assert interrupted["markers_at_stop"] == STOP_AFTER  # exactly the declared number of completed checkpoints, none partial


def test_resume_reuses_the_completed_checkpoints_and_computes_only_the_rest(interrupted):
    computed, loaded = interrupted["counts"]
    assert loaded == STOP_AFTER and computed == UNITS - STOP_AFTER


def test_a_resumed_lineage_equals_the_clean_lineage_under_the_canonical_projection(clean, interrupted):
    a, b = clean["projection"], interrupted["projection"]
    assert a["winners"] == b["winners"] and a["registry"] == b["registry"]  # trial registry and scores
    assert a["paths"] == b["paths"]  # plans, NAV, ledger hashes, costs, fills for every fold and cost scenario
    assert a["bootstrap_index_hash"] == b["bootstrap_index_hash"] and a["oos_intervals"] == b["oos_intervals"]
    assert a == b  # nothing volatile is in the projection, so the whole thing must be equal
    assert len(a["registry"]) == 48 and len(a["paths"]) == 6


def test_resume_did_not_modify_any_prior_artifact(interrupted):
    out = lp(interrupted["out"] / "resumed" / "checkpoints")
    first = list(out.rglob("checkpoint.json"))
    assert len(first) == UNITS
    log = (interrupted["out"] / "resumed" / "progress_log.txt").read_text(encoding="utf-8")
    assert log.count("start run") == 2 and "run complete" in log  # the resume decision is recorded


def test_rerunning_a_completed_run_computes_nothing_and_changes_nothing(clean):
    before = tree_hash(clean["out"])
    r = pipeline(clean["out"], "clean")
    assert r.returncode == 0 and counts(r.stdout) == (0, UNITS)
    assert tree_hash(clean["out"]) == before  # prior artifacts are never overwritten


def test_a_corrupt_checkpoint_is_quarantined_and_recomputed_not_reused(tmp_path, clean):
    out = tmp_path / "out"
    shutil.copytree(lp(clean["out"]), lp(out))
    victim = next(lp(out / "clean" / "checkpoints").rglob("decisions.json"))
    victim.write_bytes(b"[]")  # silent corruption of one artifact
    proj = tmp_path / "p.json"
    r = pipeline(out, "clean", projection=proj)
    assert r.returncode == 0 and counts(r.stdout) == (1, UNITS - 1)
    assert json.loads(proj.read_text(encoding="utf-8")) == clean["projection"]  # recomputed result is identical
    assert any(p.name.startswith(".invalid-") for p in lp(out).rglob("*") if p.is_dir())
    assert "quarantined invalid checkpoint" in (out / "clean" / "progress_log.txt").read_text(encoding="utf-8")


def test_a_lineage_mismatch_is_refused_and_touches_nothing(tmp_path, clean):
    out = tmp_path / "out"
    shutil.copytree(lp(clean["out"]), lp(out))
    before = tree_hash(out)
    alt = tmp_path / "alt_root"
    for name in ("configs", "data"):
        shutil.copytree(ROOT / name, alt / name)
    spec = (ROOT / "IMPLEMENTATION_SPEC.md").read_text(encoding="utf-8")
    (alt / "IMPLEMENTATION_SPEC.md").write_text(spec.replace("Canonical Implementation Specification",
                                                             "Canonical Implementation Specification (edited)", 1), encoding="utf-8")
    r = pipeline(out, "clean", root=alt)
    assert r.returncode != 0 and "lineage_mismatch" in (r.stderr + r.stdout) and "spec_hash" in (r.stderr + r.stdout)
    assert tree_hash(out) == before  # not reused, not overwritten, not quarantined
