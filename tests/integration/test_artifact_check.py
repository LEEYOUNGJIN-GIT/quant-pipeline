"""The artifact verifier accepts an intact, sealed research run and rejects tampering, a missing report format and a holdout artifact without an audit."""
from __future__ import annotations

import shutil

import pytest

from quant_pipeline.artifact_check import check_run, compare_runs


@pytest.fixture()
def copy(research_base, tmp_path):
    dest = tmp_path / "r1"
    shutil.copytree(research_base["out"] / "r1", dest, symlinks=True)
    return dest


def failed(report):
    return {c["name"] for c in report["checks"] if not c["ok"]}


def test_an_intact_sealed_run_passes_every_check(copy):
    report = check_run(copy)
    assert report["passed"], [c for c in report["checks"] if not c["ok"]]
    names = {c["name"] for c in report["checks"]}
    assert {"every_checkpoint_intact", "single_lineage", "evidence_class_consistent", "holdout_sealed", "pre_holdout_report_passes_terminology_scan"} <= names


def test_a_single_changed_byte_is_found(copy):
    victim = min(copy.rglob("strategy.nav.json"))
    victim.write_bytes(victim.read_bytes().replace(b"0", b"1", 1))
    assert "every_checkpoint_intact" in failed(check_run(copy))


def test_a_missing_report_format_and_a_dirty_terminology_scan_are_found(copy):
    html = next(copy.rglob("report.html"))
    html.unlink()
    assert "report_in_both_formats" in failed(check_run(copy)) or "every_checkpoint_intact" in failed(check_run(copy))


def test_a_holdout_artifact_without_an_audit_record_breaks_the_seal(copy):
    src = next(p.parent for p in copy.rglob("checkpoint.json") if "fold_id=outer-1-x1" in p.as_posix())
    shutil.copytree(src, copy / "checkpoints" / "fold_id=holdout-x1" / "trial_hash=copy" / "stage=11_oos_evaluation" / "refit_id=none")
    assert "holdout_sealed" in failed(check_run(copy))


def test_run_comparison_is_byte_exact(research_base, copy):
    assert compare_runs(research_base["out"] / "r1", copy)["identical"]
    victim = min(copy.rglob("strategy.nav.json"))
    victim.write_bytes(victim.read_bytes() + b" ")
    cmp = compare_runs(research_base["out"] / "r1", copy)
    assert not cmp["identical"] and cmp["n_different"] == 1
