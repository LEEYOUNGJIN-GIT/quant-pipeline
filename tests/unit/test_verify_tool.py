"""tools/verify.py negative controls (INV-VERIFY-001): a green report needs real evidence for every check."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import verify

SPEC = [("INV-A-001", "V0"), ("INV-B-001", "V1")]
ENTRY = {"tier": "V0", "scope": "kernel_unit", "tests": ["tests/unit/test_x.py"], "gaps": []}
REG = {"INV-A-001": dict(ENTRY), "INV-B-001": {**ENTRY, "tier": "V1"}}
PASS = {"tests/unit/test_x.py::test_a": "passed", "tests/unit/test_x.py::test_b": "passed"}


def report(results, registry=REG, spec=SPEC):
    return verify.build_report(results, registry, spec, "t", run_specials=False)


def test_all_evidence_passing_gives_overall_pass():
    r = report(PASS)
    assert r["overall"] == "pass" and r["summary"]["by_status"] == {"pass": 2}


def test_a_failed_test_fails_the_check_and_the_report():
    r = report({**PASS, "tests/unit/test_x.py::test_b": "failed"})
    assert r["overall"] == "fail" and {c["status"] for c in r["checks"]} == {"fail"}


def test_a_skipped_mandatory_test_is_not_a_pass():
    assert report({**PASS, "tests/unit/test_x.py::test_b": "skipped"})["overall"] == "fail"


def test_a_check_without_evidence_is_not_covered():
    reg = {**REG, "INV-B-001": {**REG["INV-B-001"], "tests": []}}
    r = report(PASS, reg)
    assert r["overall"] == "fail" and r["summary"]["by_status"]["not_covered"] == 1


def test_a_registry_pattern_matching_no_test_is_a_registry_error():
    reg = {**REG, "INV-B-001": {**REG["INV-B-001"], "tests": ["tests/unit/test_missing.py"]}}
    assert report(PASS, reg)["summary"]["by_status"]["registry_error"] == 1


def test_a_recorded_gap_downgrades_a_green_check_to_partial():
    reg = {**REG, "INV-B-001": {**REG["INV-B-001"], "gaps": ["something is missing"]}}
    r = report(PASS, reg)
    assert r["overall"] == "fail" and r["summary"]["by_status"] == {"partial": 1, "pass": 1}


def test_a_strict_xfail_is_a_known_gap_when_the_registry_says_so():
    reg = {**REG, "INV-B-001": {**REG["INV-B-001"], "xfail_is_gap": True}}
    r = report({**PASS, "tests/unit/test_x.py::test_b": "xfail"}, reg)
    assert r["summary"]["by_status"] == {"known_gap": 1, "pass": 1} and r["overall"] == "fail"


def test_registry_must_cover_exactly_the_spec_ids_and_tiers():
    assert report(PASS, {"INV-A-001": REG["INV-A-001"]})["registry_problems"][0]["registry_missing"] == ["INV-B-001"]
    assert report(PASS, {**REG, "INV-Z-001": ENTRY})["overall"] == "fail"
    assert report(PASS, {**REG, "INV-B-001": {**REG["INV-B-001"], "tier": "V3"}})["registry_problems"]


def test_the_shipped_registry_matches_the_spec_table_exactly():
    spec = verify.spec_check_ids((ROOT / "IMPLEMENTATION_SPEC.md").read_text(encoding="utf-8"))
    reg = verify.load_registry()
    assert len(spec) == 54 and {i for i, _ in spec} == set(reg) and all(reg[i]["tier"] == t for i, t in spec)


def test_the_report_is_written_atomically_and_leaves_no_temp_file(tmp_path):
    out = tmp_path / "sub" / "verify_report.json"
    verify.write_atomic(out, report(PASS))
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == verify.REPORT_SCHEMA
    assert [p.name for p in out.parent.iterdir()] == ["verify_report.json"]


def test_junit_parsing_maps_pass_fail_skip_and_xfail(tmp_path):
    xml = ('<testsuites><testsuite><testcase classname="tests.unit.test_verify_tool" name="a"/>'
           '<testcase classname="tests.unit.test_verify_tool" name="b"><failure message="x"/></testcase>'
           '<testcase classname="tests.unit.test_verify_tool" name="c"><skipped type="pytest.skip" message="s"/></testcase>'
           '<testcase classname="tests.unit.test_verify_tool" name="d"><skipped type="pytest.xfail" message="xfail"/></testcase>'
           '</testsuite></testsuites>')
    f = tmp_path / "j.xml"
    f.write_text(xml, encoding="utf-8")
    got = verify.parse_junit(f)
    base = "tests/unit/test_verify_tool.py::"
    assert got == {base + "a": "passed", base + "b": "failed", base + "c": "skipped", base + "d": "xfail"}


def test_pattern_matching_requires_a_prefix_on_the_test_name_or_the_file():
    r = {"tests/unit/a.py::test_one": "passed", "tests/unit/ab.py::test_one": "failed"}
    assert verify.match(r, "tests/unit/a.py") == {"tests/unit/a.py::test_one": "passed"}  # no accidental prefix collision
    assert verify.match(r, "tests/unit/a.py::test_o") == {"tests/unit/a.py::test_one": "passed"}


@pytest.mark.parametrize("status", ["pass", "fail"])
def test_exit_status_follows_overall(status):
    r = report(PASS if status == "pass" else {**PASS, "tests/unit/test_x.py::test_a": "failed"})
    assert (r["overall"] == "pass") == (status == "pass")


def test_a_lock_candidate_run_never_reports_the_lock_as_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_LOCK_CANDIDATE", "1")
    ok, note = verify.special_accepted_lock()
    assert ok is False and "CANDIDATE" in note
    monkeypatch.delenv("QUANT_LOCK_CANDIDATE")
    assert verify.special_accepted_lock()[0] is (ROOT / "uv.lock").exists()


def _gap_report(results, registry):
    return verify.build_report(results, registry, SPEC, "t", run_specials=False)


GAP_REG = {"INV-A-001": {**ENTRY, "gaps": ["not implemented"]}, "INV-B-001": {**ENTRY, "tier": "V1"}}
ALLOWED = {"INV-A-001": "declared"}


def test_the_known_gap_gate_accepts_only_pinned_partial_checks():
    r = _gap_report(PASS, GAP_REG)
    assert r["overall"] == "fail"  # strict verdict is unchanged
    g = verify.known_gap_gate(r, ALLOWED)
    assert g["ok"] and g["known_gaps_remaining"] == ["INV-A-001"] and g["unexpected_partial"] == []


def test_the_known_gap_gate_rejects_an_unpinned_gap_a_failure_and_a_skip():
    assert not verify.known_gap_gate(_gap_report(PASS, GAP_REG), {})["ok"]  # new gap not in the pinned list
    failed = _gap_report({**PASS, "tests/unit/test_x.py::test_a": "failed"}, GAP_REG)
    assert not verify.known_gap_gate(failed, ALLOWED)["ok"]
    skipped = _gap_report({**PASS, "tests/unit/test_x.py::test_a": "skipped"}, GAP_REG)
    assert not verify.known_gap_gate(skipped, ALLOWED)["ok"]
    assert not verify.known_gap_gate(_gap_report(PASS, {**GAP_REG, "INV-B-001": {**GAP_REG["INV-B-001"], "tests": []}}), ALLOWED)["ok"]  # not covered


def test_the_known_gap_gate_reports_pins_that_now_pass():
    g = verify.known_gap_gate(_gap_report(PASS, REG), ALLOWED)
    assert g["ok"] and g["pinned_but_now_passing"] == ["INV-A-001"]


def test_the_shipped_pins_match_the_shipped_registry_gaps():
    registry = verify.load_registry()
    declared = {i for i, e in registry.items() if e.get("gaps")}
    assert declared == set(verify.load_known_gaps())  # a gap added to the registry must be pinned (reviewed) on purpose


GIT = shutil.which("git")


def _repo(tmp_path):
    subprocess.run([GIT, "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x", encoding="utf-8")
    subprocess.run([GIT, "add", "f"], cwd=tmp_path, check=True)
    subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "m"], cwd=tmp_path, check=True)
    return subprocess.run([GIT, "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(GIT is None, reason="git is required")
def test_ci_provenance_passes_only_for_a_verified_ci_run(tmp_path):
    head = _repo(tmp_path)
    env = {"GITHUB_ACTIONS": "true", "GITHUB_SHA": head, "GITHUB_RUN_ID": "7", "GITHUB_REPOSITORY": "o/r", "GITHUB_WORKFLOW": "w"}
    ok, note = verify.special_ci_provenance(tmp_path, env)
    assert ok and "verified" in note and "run_id=7" in note
    assert not verify.special_ci_provenance(tmp_path, {**env, "GITHUB_SHA": "b" * 40})[0]  # runner SHA differs from HEAD
    assert not verify.special_ci_provenance(tmp_path, {k: v for k, v in env.items() if k != "GITHUB_RUN_ID"})[0]
    assert not verify.special_ci_provenance(tmp_path, {k: v for k, v in env.items() if k != "GITHUB_REPOSITORY"})[0]


@pytest.mark.skipif(GIT is None, reason="git is required")
def test_ci_provenance_is_explicitly_unavailable_in_a_local_run_and_never_fabricated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _repo(repo)
    ok, note = verify.special_ci_provenance(repo, {"GITHUB_SHA": head, "GITHUB_RUN_ID": "7", "GITHUB_REPOSITORY": "o/r"})  # no GITHUB_ACTIONS
    assert not ok and "not a CI run" in note and "unavailable" in note
    bare = tmp_path / "not_a_repo"  # a sibling, not inside the repository
    bare.mkdir()  # CI-looking environment, but the directory has no HEAD to compare with: unverifiable, not verified
    assert not verify.special_ci_provenance(bare, {"GITHUB_ACTIONS": "true", "GITHUB_SHA": head, "GITHUB_RUN_ID": "7", "GITHUB_REPOSITORY": "o/r"})[0]
