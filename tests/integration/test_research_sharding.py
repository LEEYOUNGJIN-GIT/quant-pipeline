"""Trial-path sharding (several runners compute disjoint trial paths, their checkpoints are merged by copying, one run finishes the rest) must
reproduce the unsharded research run exactly. SYNTHETIC reduced configuration, engineering evidence only."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quant_pipeline.research_pipeline import ResearchPipeline

ROOT = Path(__file__).resolve().parents[2]


def artifacts(checkpoints: Path) -> dict[str, bytes]:
    return {str(p.relative_to(checkpoints)): p.read_bytes() for p in checkpoints.rglob("*")
            if p.is_file() and p.name not in ("checkpoint.json", "COMPLETE") and not any(x.startswith(".") for x in p.parts)}


@pytest.fixture(scope="module")
def merged(research_base, tmp_path_factory):
    base = tmp_path_factory.mktemp("shards")
    for i in (0, 1):
        r = subprocess.run([sys.executable, "-m", "quant_pipeline.research_pipeline", "--root", str(research_base["root"]), "--out-root", str(base / f"s{i}"),
                            "--config", research_base["cfg"], "--run-id", "r1", "--allow-synthetic", "--trial-shard", f"{i}/2"],
                           capture_output=True, text=True, cwd=ROOT, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
    out = base / "merged"
    for i in (0, 1):
        shutil.copytree(base / f"s{i}" / "r1" / "checkpoints", out / "r1" / "checkpoints", dirs_exist_ok=True)
    p = ResearchPipeline(research_base["root"], out, "r1", config=research_base["cfg"], allow_synthetic=True)
    return {"p": p, "proj": p.run(), "out": out, "shards": [base / "s0", base / "s1"]}


def test_each_shard_writes_only_its_own_trial_paths(merged, research_base):
    counts = []
    for s in merged["shards"]:
        sel = sorted(d.name for d in (s / "r1" / "checkpoints" / "fold_id=selection").iterdir())
        counts.append(len(sel))
        assert not list((s / "r1" / "checkpoints").glob("fold_id=outer*")) and not (s / "r1" / "checkpoints" / "fold_id=final").exists()
    assert counts == [2, 2]  # the reduced design has four trials: positions 0 and 2, then 1 and 3


def test_the_finishing_run_loads_every_shard_checkpoint_and_computes_none_of_the_trial_paths(merged):
    p = merged["p"]
    assert len([x for x in p.loaded if "|selection|" in x and "10_event_time" in x]) == 4
    assert not [x for x in p.computed if "|selection|" in x and "10_event_time" in x]


def test_the_merged_run_equals_the_unsharded_run_exactly(merged, research_base):
    assert merged["proj"] == research_base["proj"]
    a, b = artifacts(research_base["out"] / "r1" / "checkpoints"), artifacts(merged["out"] / "r1" / "checkpoints")
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)
