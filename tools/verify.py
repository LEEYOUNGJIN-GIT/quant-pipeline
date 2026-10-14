#!/usr/bin/env python3
"""Single verification orchestrator (spec 17.1, check ``INV-VERIFY-001``).

Runs the test suite once (or reuses a junit file), maps results onto every §17.2 check ID through ``tests/check_registry.py`` and
writes ``outputs/verify_report.json`` atomically. A check passes only if all its evidence passed and it has no recorded gap.
Exit code 0 only when every mandatory check passes; failed, skipped, partial, known-gap and uncovered checks all fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

REPORT_SCHEMA = "verify-report-1"
PASSING = {"pass"}


# ------------------------------------------------------------------------------------------------ spec / registry
def spec_check_ids(spec_text: str) -> list[tuple[str, str]]:
    block = spec_text.split("### 17.2", 1)[1].split("### 17.3", 1)[0]
    rows = re.findall(r"\|[^|]*\|\s*`(INV-[A-Z]+-\d+)`\s*\|\s*(V\d)\s*\|", block)
    return rows


def load_registry() -> dict[str, dict[str, Any]]:
    import importlib

    return importlib.import_module("check_registry").REGISTRY


# ------------------------------------------------------------------------------------------------ junit
def parse_junit(path: Path) -> dict[str, str]:
    """nodeid -> passed | failed | skipped | xfail."""
    out: dict[str, str] = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        file = tc.get("classname", "").replace(".", "/")
        # a class-based test has ``module.Class`` as classname; keep only the module part that exists as a file
        while file and not (ROOT / f"{file}.py").exists() and "/" in file:
            file = file.rsplit("/", 1)[0]
        node = f"{file}.py::{tc.get('name', '')}"
        status = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                status = "failed"
            elif child.tag == "skipped":
                status = "xfail" if "xfail" in (child.get("type", "") + child.get("message", "")).lower() else "skipped"
        out[node] = status
    return out


def match(results: dict[str, str], pattern: str) -> dict[str, str]:
    if "::" in pattern:
        file, prefix = pattern.split("::", 1)
        return {n: s for n, s in results.items() if n.startswith(f"{file}::") and n.split("::", 1)[1].startswith(prefix)}
    return {n: s for n, s in results.items() if n.startswith(f"{pattern}::")}


# ------------------------------------------------------------------------------------------------ special evidence
def special_accepted_lock() -> tuple[bool, str]:
    lock = ROOT / "uv.lock"
    if not lock.exists():
        return False, "no accepted uv.lock: the environment is not frozen (spec 2.2 accepts the lock only after target-VM verification)"
    return True, "uv.lock present: " + "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()


def special_spec_lint() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "spec_lint.py"), "--output", str(Path(tempfile.mkdtemp()) / "lint.json")],
                       capture_output=True, text=True, cwd=ROOT, check=False)
    return r.returncode == 0, f"spec_lint exit {r.returncode}"


def special_after_plan_gate() -> tuple[bool, str]:
    from quant_pipeline.calendars import Calendars, build_row_map
    from quant_pipeline.config import load_config
    from quant_pipeline.fitstate import lineage_hashes
    from quant_pipeline.plan import build_compute_plan, doctor_after_plan
    from quant_pipeline.splits import build_split_manifest
    from quant_pipeline.trials import enumerate_trials

    cfg = load_config(ROOT / "configs" / "fixture_3asset_aux1024.yaml")
    cals = Calendars.load(["XKRX", "XNYS"])
    rm, _ = build_row_map(cfg, cals)
    plan = build_compute_plan(cfg, build_split_manifest(cfg, cals, rm), enumerate_trials(cfg), lineage_hashes(cfg, ROOT))
    tmp = Path(tempfile.mkdtemp())
    (tmp / "v").mkdir()
    rep = doctor_after_plan(cfg, plan, tmp / "v" / "compute_plan.json", tmp, "v")
    return rep["status"] == "pass", f"doctor --after-plan on the revised fixture profile (D6, OQ-032: auxiliary ceiling 1024): {rep['status']}" + (
        f" (failed: {', '.join(rep['failed'])})" if rep["failed"] else "")


SPECIALS = {"accepted_lock": special_accepted_lock, "spec_lint": special_spec_lint, "after_plan_gate_on_shipped_profile": special_after_plan_gate}


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(check_id: str, entry: dict[str, Any], results: dict[str, str], run_specials: bool = True) -> dict[str, Any]:
    matched: dict[str, str] = {}
    unmatched_patterns = []
    for pat in entry["tests"]:
        m = match(results, pat)
        if not m:
            unmatched_patterns.append(pat)
        matched.update(m)
    failed = sorted(n for n, s in matched.items() if s == "failed")
    skipped = sorted(n for n, s in matched.items() if s == "skipped")
    xfail = sorted(n for n, s in matched.items() if s == "xfail")
    passed = sum(1 for s in matched.values() if s == "passed")
    special_notes, special_ok = [], True
    if run_specials:
        for name in entry.get("special", []):
            if name in SPECIALS:
                ok, note = SPECIALS[name]()
                special_ok &= ok
                special_notes.append(f"{name}: {note}")
    reasons = []
    if not entry["tests"] and not entry.get("special"):
        status = "not_covered"
        reasons.append("no evidence exists for this check")
    elif unmatched_patterns:
        status = "registry_error"
        reasons.append(f"registry patterns matched no test: {unmatched_patterns}")
    elif failed:
        status = "fail"
        reasons.append(f"{len(failed)} evidence test(s) failed")
    elif skipped:
        status = "fail"
        reasons.append("a mandatory evidence test was skipped")
    elif not special_ok:
        status = "fail"
        reasons.append("special evidence failed")
    elif xfail and entry.get("xfail_is_gap"):
        status = "known_gap"
        reasons.append("a strict-xfail test records a known acceptance failure")
    elif entry.get("gaps"):
        status = "partial"
        reasons.append("evidence is incomplete against the spec's minimum-evidence column")
    else:
        status = "pass"
    return {"id": check_id, "tier": entry["tier"], "evidence_scope": entry["scope"], "mandatory": True, "status": status,
            "tests_matched": len(matched), "tests_passed": passed, "failed_tests": failed, "skipped_tests": skipped, "xfail_tests": xfail,
            "gaps": entry.get("gaps", []), "special": special_notes, "reasons": reasons}


def lineage() -> dict[str, str]:
    from quant_pipeline.config import load_config
    from quant_pipeline.fitstate import lineage_hashes

    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    h = lineage_hashes(cfg, ROOT)
    lock = ROOT / "uv.lock"
    return {"spec_hash": h["spec_hash"], "config_hash": h["base_profile_hash"], "data_manifest_hash": h["data_manifest_hash"],
            "code_hash": h["code_hash"], "lock_hash": ("sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()) if lock.exists()
            else "unavailable:no_accepted_lock"}


def build_report(results: dict[str, str], registry: dict[str, dict[str, Any]], spec_ids: list[tuple[str, str]], run_id: str,
                 tiers: set[str] | None = None, run_specials: bool = True) -> dict[str, Any]:
    spec_set = {i for i, _ in spec_ids}
    problems = []
    if set(registry) != spec_set:
        problems.append({"registry_missing": sorted(spec_set - set(registry)), "registry_unknown": sorted(set(registry) - spec_set)})
    tier_of = dict(spec_ids)
    wrong_tier = sorted(i for i in registry if i in tier_of and registry[i]["tier"] != tier_of[i])
    if wrong_tier:
        problems.append({"tier_mismatch": wrong_tier})
    checks = [evaluate(i, e, results, run_specials) for i, e in registry.items() if i in spec_set and (tiers is None or e["tier"] in tiers)]
    counts: dict[str, int] = {}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    by_scope: dict[str, int] = {}
    for c in checks:
        if c["status"] == "pass":
            by_scope[c["evidence_scope"]] = by_scope.get(c["evidence_scope"], 0) + 1
    mandatory_not_passing = sorted(c["id"] for c in checks if c["mandatory"] and c["status"] not in PASSING)
    overall = "pass" if not problems and not mandatory_not_passing and checks else "fail"
    return {"schema": REPORT_SCHEMA, "run_id": run_id, "overall": overall, "lineage": lineage(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "spec_target_python": "3.12"},
            "summary": {"checks": len(checks), "by_status": dict(sorted(counts.items())), "passing_by_evidence_scope": by_scope,
                        "mandatory_not_passing": mandatory_not_passing, "tests_seen": len(results),
                        "tests_failed": sum(1 for s in results.values() if s == "failed"),
                        "tests_skipped": sum(1 for s in results.values() if s == "skipped"),
                        "tests_xfail": sum(1 for s in results.values() if s == "xfail")},
            "registry_problems": problems, "checks": checks,
            "evidence_labels": "independent_oracle = compared with tools/oracle_fixture.py or tools/hmm_probe_oracle.py (kernels only); "
                               "reviewed_regression = production-dependent snapshot, never an independent oracle; kernel_unit = explicit inputs only"}


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_pytest(junit: Path, extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *extra]
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="verify")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "verify_report.json")
    ap.add_argument("--junit", type=Path, default=None, help="reuse an existing junit XML instead of running pytest")
    ap.add_argument("--tier", action="append", choices=["V0", "V1", "V2", "V3", "V4"], help="limit the report to these tiers")
    ap.add_argument("pytest_args", nargs="*")
    args = ap.parse_args()
    junit = args.junit
    if junit is None:
        junit = Path(tempfile.mkdtemp()) / "junit.xml"
        run_pytest(junit, args.pytest_args)
    if not junit.exists():
        print("verification failed: no junit results were produced", file=sys.stderr)
        return 2
    spec = (ROOT / "IMPLEMENTATION_SPEC.md").read_text(encoding="utf-8")
    report = build_report(parse_junit(junit), load_registry(), spec_check_ids(spec), args.run_id, set(args.tier) if args.tier else None)
    report["generated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")  # volatile; excluded from any comparison
    write_atomic(args.output, report)
    s = report["summary"]
    print(f"verify: {report['overall'].upper()} - {s['checks']} checks {s['by_status']}; tests failed {s['tests_failed']}, "
          f"skipped {s['tests_skipped']}, xfail {s['tests_xfail']}")
    for c in report["checks"]:
        if c["status"] != "pass":
            print(f"  {c['status']:13s} {c['id']:18s} {c['tier']}  {'; '.join(c['reasons'] + c['special'])[:150]}")
    print(f"report: {args.output}")
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
