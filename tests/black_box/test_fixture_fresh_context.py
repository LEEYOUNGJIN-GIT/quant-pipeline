"""Fresh-context, network-denied fixture replay (spec 17.3, V4 checks INV-BLACKBOX-001 / INV-CHK-002).

The child processes run in a new temporary root that holds only package source, specification, profiles, fixture inputs, checked-in
expected files and the oracle scripts. Network denial is a Python-level hook injected through ``sitecustomize`` and proven active by a
preliminary child that must be refused before anything else runs. Nothing here regenerates expected files.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from quant_pipeline.config import load_config
from quant_pipeline.fitstate import lineage_hashes

ROOT = Path(__file__).resolve().parents[2]
DENY_DIR = Path(__file__).resolve().parent / "netdeny"
PROFILE = "configs/fixture_3asset_aux1024.yaml"
EXPECTED = ROOT / "tests" / "expected" / "fixture_3asset" / "regression"
SECRET_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "GITHUB_TOKEN", "GH_TOKEN",
               "GITHUB_ACTIONS", "GITHUB_SHA", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
PROBE = (
    "import socket\n"
    "def attempt(f):\n"
    "    try:\n"
    "        f()\n"
    "    except Exception as e:\n"
    "        return type(e).__name__\n"
    "    return 'CONNECTED'\n"
    "print('DNS', attempt(lambda: socket.getaddrinfo('example.com', 80)))\n"
    "print('TCP', attempt(lambda: socket.create_connection(('127.0.0.1', 9), timeout=2)))\n"
    "print('CONNECT', attempt(lambda: socket.socket().connect(('127.0.0.1', 9))))\n"
)


def child_env(src: Path, *, deny: bool = True) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SECRET_VARS}
    env["PYTHONPATH"] = os.pathsep.join([str(DENY_DIR)] * deny + [str(src)])
    env["PYTHONHASHSEED"] = "0"
    return env


def denial_proven(cwd: Path, env: dict[str, str]) -> bool:
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=60)
    got = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    return r.returncode == 0 and got == {"DNS": "NetworkDenied", "TCP": "NetworkDenied", "CONNECT": "NetworkDenied"}


def lp(p: Path) -> Path:
    prefix = "\\\\?\\"
    s = str(Path(p).resolve())
    return Path(prefix + s) if sys.platform == "win32" and not s.startswith(prefix) else Path(s)


def build_root(dest: Path) -> Path:
    """Copy only what a fresh context may use; prior outputs, caches, checkpoints and local overrides are excluded."""
    dest.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", dest / "src", ignore=ignore)
    shutil.copytree(ROOT / "configs", dest / "configs")
    shutil.copytree(ROOT / "data" / "fixtures", dest / "data" / "fixtures")
    shutil.copytree(ROOT / "tests" / "expected", dest / "tests" / "expected")
    (dest / "tools").mkdir()
    shutil.copy(ROOT / "tools" / "oracle_fixture.py", dest / "tools" / "oracle_fixture.py")
    shutil.copy(ROOT / "IMPLEMENTATION_SPEC.md", dest / "IMPLEMENTATION_SPEC.md")
    for f in (dest / "tests" / "expected").rglob("*"):  # expected files are read-only in the fresh context
        if f.is_file():
            f.chmod(stat.S_IREAD)
    return dest


def cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "quant_pipeline.cli", *args, "--repo-root", str(root)], cwd=root,
                          env=child_env(root / "src"), capture_output=True, text=True, check=False, timeout=1800)


def pipeline(root: Path, out: Path, run_id: str, proj: Path | None = None, stop_after: int | None = None):
    cmd = [sys.executable, "-m", "quant_pipeline.pipeline", "--run-id", run_id, "--out-root", str(out), "--root", str(root), "--config", PROFILE]
    if stop_after is not None:
        cmd += ["--stop-after", str(stop_after)]
    if proj is not None:
        cmd += ["--projection-out", str(proj)]
    return subprocess.run(cmd, cwd=root, env=child_env(root / "src"), capture_output=True, text=True, check=False, timeout=1800)


def compare_projection(actual: dict, expected: dict, spec: dict, lineage: dict) -> list[str]:
    """Strict: every field is compared; exclusions come only from the checked-in spec and name exact top-level keys."""
    assert all(isinstance(x, str) and "*" not in x for x in spec["excluded_fields"])  # no wildcard/regex exclusions
    a = {k: v for k, v in actual.items() if k not in spec["excluded_fields"]}
    got_lineage = a.pop("lineage")
    problems = [f"lineage.{k}" for k in sorted(set(lineage) | set(got_lineage)) if lineage.get(k) != got_lineage.get(k)]
    e = {k: v for k, v in expected.items() if k not in spec["excluded_fields"]}
    problems += [f"unexpected field {k}" for k in sorted(set(a) - set(e))] + [f"missing field {k}" for k in sorted(set(e) - set(a))]
    problems += [f"differs: {k}" for k in sorted(set(a) & set(e)) if json.loads(json.dumps(a[k])) != e[k]]
    return problems


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("bb")
    root = build_root(base / "fresh")
    other = build_root(base / "fresh2")
    return {"base": base, "root": root, "other": other}


@pytest.fixture(scope="module")
def lineage():
    """Freshly computed for the verified tree (not stored in the snapshot), in the pipeline's lineage vocabulary."""
    h = lineage_hashes(load_config(ROOT / PROFILE), ROOT)
    lock = ROOT / "uv.lock"
    return {"spec_hash": h["spec_hash"], "config_hash": h["base_profile_hash"], "data_manifest_hash": h["data_manifest_hash"],
            "lock_hash": "unavailable:no_accepted_lock" if not lock.exists() else "unchecked", "code_hash": h["code_hash"]}


def test_the_denial_hook_is_proven_active_and_a_missing_hook_is_detected(world):
    root = world["root"]
    assert denial_proven(root, child_env(root / "src"))
    assert not denial_proven(root, child_env(root / "src", deny=False))  # negative control: without the hook the probe is not refused


def test_the_fresh_context_contains_only_permitted_inputs_and_imports_its_own_source(world):
    root = world["root"]
    names = {p.name for p in root.iterdir()}
    assert names == {"src", "configs", "data", "tests", "tools", "IMPLEMENTATION_SPEC.md"}  # no outputs, caches, notebooks or overrides
    r = subprocess.run([sys.executable, "-c", "import quant_pipeline,sys;print(quant_pipeline.__file__)"], cwd=root,
                       env=child_env(root / "src"), capture_output=True, text=True, check=False)
    assert Path(r.stdout.strip()).resolve().is_relative_to(root.resolve())
    assert not os.access(root / "tests" / "expected" / "fixture_3asset" / "regression" / "projection.json", os.W_OK)


def test_static_doctor_plan_and_after_plan_doctor_then_the_gated_run_all_succeed_offline(world, lineage):
    root = world["root"]
    cfg = ["--config", str(root / PROFILE)]
    for step in (["doctor", *cfg], ["plan", *cfg, "--run-id", "bb"], ["doctor", *cfg, "--after-plan", "--run-id", "bb"]):
        r = cli(root, *step)
        assert r.returncode == 0, f"{step[0]} failed:\n{r.stdout}{r.stderr}"
    r = cli(root, "run", *cfg, "--run-id", "bb")
    assert r.returncode == 0 and "run complete" in r.stdout, r.stdout + r.stderr
    proj = json.loads((root / "outputs" / "runs" / "bb" / "projection.json").read_text(encoding="utf-8"))
    spec = json.loads((EXPECTED / "projection_spec.json").read_text(encoding="utf-8"))
    expected = json.loads((EXPECTED / "projection.json").read_text(encoding="utf-8"))
    assert compare_projection(proj, expected, spec, lineage) == []  # reviewed_regression


def test_two_clean_runs_and_an_interrupted_resumed_run_are_identical_and_match_the_snapshot(world, lineage):
    base, root, other = world["base"], world["root"], world["other"]
    spec = json.loads((EXPECTED / "projection_spec.json").read_text(encoding="utf-8"))
    expected = json.loads((EXPECTED / "projection.json").read_text(encoding="utf-8"))
    p1, p2, p3 = base / "p1.json", base / "p2.json", base / "p3.json"
    assert pipeline(root, base / "o1", "c1", p1).returncode == 0
    assert pipeline(other, base / "o2", "c2", p2).returncode == 0  # separate directory, separate copy of the tree
    stopped = pipeline(root, base / "o3", "r1", stop_after=5)
    assert stopped.returncode == 75 and "STOPPED" in stopped.stdout
    resumed = pipeline(root, base / "o3", "r1", p3)  # a new process resumes the same lineage
    assert resumed.returncode == 0 and "loaded=5" in resumed.stdout
    a, b, c = (json.loads(p.read_text(encoding="utf-8")) for p in (p1, p2, p3))
    assert a == b == c  # clean-versus-clean and clean-versus-resume; no run id is part of the projection
    assert compare_projection(a, expected, spec, lineage) == []


def test_the_comparison_rejects_changed_unknown_and_wrongly_excluded_fields(lineage):
    spec = json.loads((EXPECTED / "projection_spec.json").read_text(encoding="utf-8"))
    expected = json.loads((EXPECTED / "projection.json").read_text(encoding="utf-8"))
    base = {**expected, "lineage": lineage}
    assert compare_projection(base, expected, spec, lineage) == []
    assert compare_projection({**base, "winners": {"x": "y"}}, expected, spec, lineage) == ["differs: winners"]
    assert compare_projection({**base, "new_field": 1}, expected, spec, lineage) == ["unexpected field new_field"]
    assert compare_projection({k: v for k, v in base.items() if k != "paths"}, expected, spec, lineage) == ["missing field paths"]
    assert compare_projection({**base, "lineage": {**lineage, "code_hash": "sha256:x"}}, expected, spec, lineage) == ["lineage.code_hash"]
    with pytest.raises(AssertionError):
        compare_projection(base, expected, {**spec, "excluded_fields": ["pa*"]}, lineage)


def test_independent_oracle_files_reproduce_exactly_in_the_fresh_context(world):
    root = world["root"]
    r = subprocess.run([sys.executable, str(root / "tools" / "oracle_fixture.py")], cwd=root, env=child_env(root / "src"),
                       capture_output=True, text=True, check=False, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    # the HMM probe oracle consumes a production probe file and is compared in tests/golden/test_hmm_pipeline.py (also offline)
