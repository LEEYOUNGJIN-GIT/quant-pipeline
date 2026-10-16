"""Fresh-context, network-denied fixture replay (spec 17.3, V4 checks INV-BLACKBOX-001 / INV-CHK-002).

The child processes run in a new temporary root that holds only package source, specification, profiles, fixture inputs, checked-in
expected files and the oracle scripts. Network denial is a Python-level hook injected through ``sitecustomize`` and proven active by a
preliminary child that must be refused before anything else runs. Nothing here regenerates expected files.
"""
from __future__ import annotations

import hashlib
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


_NETNS_PROBE = "import socket,sys\ntry:\n    socket.create_connection(('192.0.2.1', 80), timeout=3)\nexcept OSError as e:\n    print(type(e).__name__)\n    sys.exit(0)\nsys.exit(3)\n"


def netns_prefix() -> tuple[list[str], str]:
    """OS-level isolation where the runner supports it (spec 17.3): ``(command prefix, "")`` or ``([], reason it is unavailable)``.

    The prefix is only returned after a child under it has been refused by the kernel itself, i.e. without the Python hook.
    """
    exe = shutil.which("unshare")
    if exe is None:
        return [], "unshare is not installed"
    prefix = [exe, "-n", "-r", "--"]
    try:
        r = subprocess.run([*prefix, sys.executable, "-c", _NETNS_PROBE], capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [], f"unshare failed to start: {type(e).__name__}"
    if r.returncode != 0:
        return [], f"unprivileged network namespaces are not permitted here (exit {r.returncode}: {r.stderr.strip()[:120] or r.stdout.strip()[:120]})"
    return prefix, ""


def build_root(dest: Path, *, with_verify: bool = False) -> Path:
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
    if (ROOT / "uv.lock").exists():  # spec 17.3: the fresh context receives the accepted lock (a candidate is flagged by the environment)
        shutil.copy(ROOT / "uv.lock", dest / "uv.lock")
    if with_verify:  # verify.py needs the checked-in test code and registry it maps onto; never the regeneration tools or prior outputs
        for sub in ("unit", "golden", "integration", "support"):
            shutil.copytree(ROOT / "tests" / sub, dest / "tests" / sub, ignore=ignore)
        for name in ("check_registry.py", "known_gaps.json"):
            shutil.copy(ROOT / "tests" / name, dest / "tests" / name)
        for name in ("verify.py", "spec_lint.py", "generate_fixture.py", "hmm_probe_oracle.py"):
            shutil.copy(ROOT / "tools" / name, dest / "tools" / name)
        for name in ("pyproject.toml", "README.md", "PROGRESS.md", "open-questions.md", "ADOPTION_MATRIX.md", "FINAL_SPEC_RECONSTRUCTION_PROMPT.md",
                     "PROJECT_ALIGNMENT_REVIEW.md", "SPEC_RECOVERY_DECISION.md"):
            if (ROOT / name).exists():
                shutil.copy(ROOT / name, dest / name)
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


def _registry_differs(a: dict, e: dict, tol: float) -> bool:
    """Entry names and eligibility flags are exact; scores may differ by at most ``tol`` (None must match None)."""
    if a.keys() != e.keys():
        return True
    for k, (score, eligible) in e.items():
        got_score, got_eligible = a[k]
        if got_eligible != eligible or (got_score is None) != (score is None):
            return True
        if score is not None and abs(got_score - score) > tol:
            return True
    return False


def _paths_differ(a: dict, e: dict) -> bool:
    """Every field is exact except ``plans``, whose hashes are compared by count and shape only (spec ``plan_hashes: count_only``)."""
    if a.keys() != e.keys():
        return True
    for key, exp in e.items():
        got = a[key]
        if got.keys() != exp.keys():
            return True
        for f, v in exp.items():
            if f == "plans":
                if len(got[f]) != len(v) or not all(isinstance(x, str) and x.startswith("sha256:") for x in got[f]):
                    return True
            elif got[f] != v:
                return True
    return False


def compare_projection(actual: dict, expected: dict, spec: dict, lineage: dict) -> list[str]:
    """Strict except for the tolerances declared in the checked-in spec; exclusions name exact top-level keys, never patterns."""
    assert all(isinstance(x, str) and "*" not in x for x in spec["excluded_fields"])  # no wildcard/regex exclusions
    tol = spec.get("tolerances", {})
    assert set(tol) <= {"registry_score_abs", "plan_hashes", "rationale"} and tol.get("plan_hashes", "count_only") == "count_only"
    a = {k: v for k, v in actual.items() if k not in spec["excluded_fields"]}
    got_lineage = a.pop("lineage")
    problems = [f"lineage.{k}" for k in sorted(set(lineage) | set(got_lineage)) if lineage.get(k) != got_lineage.get(k)]
    e = {k: v for k, v in expected.items() if k not in spec["excluded_fields"]}
    problems += [f"unexpected field {k}" for k in sorted(set(a) - set(e))] + [f"missing field {k}" for k in sorted(set(e) - set(a))]
    for k in sorted(set(a) & set(e)):
        got = json.loads(json.dumps(a[k]))
        if k == "registry" and "registry_score_abs" in tol:
            differs = _registry_differs(got, e[k], float(tol["registry_score_abs"]))
        elif k == "paths" and tol.get("plan_hashes") == "count_only":
            differs = _paths_differ(got, e[k])
        else:
            differs = got != e[k]
        if differs:
            problems.append(f"differs: {k}")
    return problems


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("bb")
    root = build_root(base / "fresh")
    other = build_root(base / "fresh2")
    return {"base": base, "root": root, "other": other, "verify_root": build_root(base / "fresh3", with_verify=True)}


@pytest.fixture(scope="module")
def lineage():
    """Freshly computed for the verified tree (not stored in the snapshot), in the pipeline's lineage vocabulary."""
    h = lineage_hashes(load_config(ROOT / PROFILE), ROOT)
    lock = ROOT / "uv.lock"  # written independently of the production helper: a candidate lock is not an accepted lock
    accepted = lock.exists() and os.environ.get("QUANT_LOCK_CANDIDATE") != "1"
    return {"spec_hash": h["spec_hash"], "config_hash": h["base_profile_hash"], "data_manifest_hash": h["data_manifest_hash"],
            "lock_hash": ("sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()) if accepted else "unavailable:no_accepted_lock", "code_hash": h["code_hash"]}


def test_the_denial_hook_is_proven_active_and_a_missing_hook_is_detected(world):
    root = world["root"]
    assert denial_proven(root, child_env(root / "src"))
    assert not denial_proven(root, child_env(root / "src", deny=False))  # negative control: without the hook the probe is not refused


def test_the_os_network_namespace_is_used_where_the_runner_supports_it_and_its_absence_is_recorded():
    prefix, reason = netns_prefix()
    if prefix:  # the kernel, not the Python hook, refuses the connection (the probe above ran without the hook)
        assert prefix[0].endswith("unshare")
    else:  # not a skip: the Python-level hook stays mandatory and the missing OS layer is stated, never silently assumed
        assert reason
    print(f"OS network namespace: {'active' if prefix else 'unavailable - ' + reason}")


def test_the_fresh_context_contains_only_permitted_inputs_and_imports_its_own_source(world):
    root = world["root"]
    names = {p.name for p in root.iterdir()}
    allowed = {"src", "configs", "data", "tests", "tools", "IMPLEMENTATION_SPEC.md"} | ({"uv.lock"} if (ROOT / "uv.lock").exists() else set())
    assert names == allowed  # no outputs, caches, notebooks or overrides
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


def test_the_declared_tolerances_accept_runner_noise_and_reject_real_changes(lineage):
    spec = json.loads((EXPECTED / "projection_spec.json").read_text(encoding="utf-8"))
    expected = json.loads((EXPECTED / "projection.json").read_text(encoding="utf-8"))
    base = {**expected, "lineage": lineage}
    key = next(iter(expected["registry"]))
    path = next(iter(expected["paths"]))

    def with_registry(entry):
        return {**base, "registry": {**expected["registry"], key: entry}}

    def with_path(**fields):
        return {**base, "paths": {**expected["paths"], path: {**expected["paths"][path], **fields}}}

    score, eligible = expected["registry"][key]
    assert compare_projection(with_registry([score + 0.017, eligible]), expected, spec, lineage) == []  # observed runner noise
    assert compare_projection(with_registry([score + 0.06, eligible]), expected, spec, lineage) == ["differs: registry"]
    assert compare_projection(with_registry([score, not eligible]), expected, spec, lineage) == ["differs: registry"]
    assert compare_projection({**base, "registry": {k: v for k, v in expected["registry"].items() if k != key}}, expected, spec, lineage) == ["differs: registry"]
    plans = expected["paths"][path]["plans"]
    assert compare_projection(with_path(plans=["sha256:" + "0" * 64] * len(plans)), expected, spec, lineage) == []  # hashes: count only
    assert compare_projection(with_path(plans=plans[:-1]), expected, spec, lineage) == ["differs: paths"]
    assert compare_projection(with_path(plans=["not-a-hash"] * len(plans)), expected, spec, lineage) == ["differs: paths"]
    for field in ("fills", "nav", "ledger_hash", "costs", "benchmarks"):
        assert compare_projection(with_path(**{field: "changed"}), expected, spec, lineage) == ["differs: paths"], field
    with pytest.raises(AssertionError):
        compare_projection(base, expected, {**spec, "tolerances": {"nav_abs": 1.0}}, lineage)


def test_independent_oracle_files_reproduce_exactly_in_the_fresh_context(world):
    root = world["root"]
    r = subprocess.run([sys.executable, str(root / "tools" / "oracle_fixture.py")], cwd=root, env=child_env(root / "src"),
                       capture_output=True, text=True, check=False, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    # the HMM probe oracle consumes a production probe file and is compared in tests/golden/test_hmm_pipeline.py (also offline)


# Research end-to-end modules (about 18 minutes together) are verified by the outer run; the fresh-context run omits them on purpose and the report must
# say so: the only check that may be left unmatched is INV-DATA-001, and only because of exactly these files.
HEAVY_RESEARCH_EVIDENCE = frozenset({"tests/integration/test_research_pipeline.py", "tests/integration/test_research_invariance.py",
                                     "tests/integration/test_research_holdout.py"})


def test_verify_py_runs_inside_the_network_denied_fresh_context_and_reports_the_same_lineage(world):
    """Spec 17.3 runs ``verify.py`` in the fresh context. The V0 evidence files are run there, except the heavy research modules named above;
    the black-box test itself is excluded (no recursion)."""
    import importlib
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tests"))
    registry = importlib.import_module("check_registry").REGISTRY
    files = sorted({p.split("::")[0] for e in registry.values() if e["tier"] == "V0" for p in e["tests"]})
    root = world["verify_root"]
    assert not any("black_box" in f for f in files)
    assert HEAVY_RESEARCH_EVIDENCE <= set(files)  # the omission below is of registered evidence, not of something that does not exist
    run_files = [f for f in files if f not in HEAVY_RESEARCH_EVIDENCE]
    prefix, _ = netns_prefix()
    out = root / "outputs" / "verify_report.json"
    cmd = [*prefix, sys.executable, str(root / "tools" / "verify.py"), "--run-id", "bb-verify", "--output", str(out), "--tier", "V0", "--allow-known-gaps", *run_files]
    r = subprocess.run(cmd, cwd=root, env=child_env(root / "src"), capture_output=True, text=True, check=False, timeout=3000)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["registry_problems"] == [] and report["checks"]
    others = {c["id"]: c for c in report["checks"] if c["status"] not in ("pass", "partial")}
    assert set(others) == {"INV-DATA-001"}, r.stdout[-3000:] + r.stderr[-3000:]  # nothing else fails, is skipped or goes unmatched
    assert others["INV-DATA-001"]["status"] == "registry_error" and others["INV-DATA-001"]["failed_tests"] == []
    unmatched = others["INV-DATA-001"]["reasons"][0]
    assert all(h in unmatched for h in HEAVY_RESEARCH_EVIDENCE) and unmatched.count("tests/") == len(HEAVY_RESEARCH_EVIDENCE)  # exactly the omitted files
    assert r.returncode == 1 and report["known_gap_gate"]["other_non_passing"] == ["INV-DATA-001"]  # the gate says so instead of passing quietly
    assert report["lineage"] == {k: v for k, v in lineage_for_verify().items()}  # the same tree was verified


def lineage_for_verify() -> dict[str, str]:
    from quant_pipeline.dependency import accepted_lock_hash

    h = lineage_hashes(load_config(ROOT / "configs" / "fixture_3asset.yaml"), ROOT)
    return {"spec_hash": h["spec_hash"], "config_hash": h["base_profile_hash"], "data_manifest_hash": h["data_manifest_hash"],
            "code_hash": h["code_hash"], "lock_hash": accepted_lock_hash(ROOT)}
