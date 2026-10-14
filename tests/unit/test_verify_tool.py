"""tools/verify.py negative controls (INV-VERIFY-001): a green report needs real evidence for every check."""
from __future__ import annotations

import json
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
