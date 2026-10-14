"""Capability probe and SHA audit (INV-GOV-001): observed facts or explicit nulls, never fabricated, local never called CI."""
from __future__ import annotations

import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from quant_pipeline import capability
from quant_pipeline.cli import app

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required for the probe tests")  # the repository itself needs git
SHA = "a" * 40


def git(root, *args):
    subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=root, check=True, capture_output=True)


def test_a_directory_that_is_not_a_repository_reports_explicit_nulls(tmp_path):
    g = capability.probe_git(tmp_path)
    assert g["is_repository"] is False and g["head_sha"] == {"value": None, "reason": "not a git work tree"}


def test_a_repository_without_a_commit_has_a_null_sha_not_a_made_up_one(tmp_path):
    git(tmp_path, "init", "-q")
    g = capability.probe_git(tmp_path)
    assert g["is_repository"] and g["head_sha"]["value"] is None and "no commit" in g["head_sha"]["reason"]
    assert g["remote_origin"]["value"] is None


def test_a_committed_repository_reports_the_real_head_and_dirty_state(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    git(tmp_path, "add", "f.txt")
    git(tmp_path, "commit", "-q", "-m", "c")
    real = subprocess.run([GIT, "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    g = capability.probe_git(tmp_path)
    assert g["head_sha"]["value"] == real and capability.SHA_RE.match(real) and g["dirty"]["value"] is False
    (tmp_path / "f.txt").write_text("changed", encoding="utf-8")
    assert capability.probe_git(tmp_path)["dirty"]["value"] is True


def test_a_local_run_is_never_described_as_ci_even_with_ci_looking_values():
    ci = capability.probe_ci({"GITHUB_SHA": SHA, "GITHUB_RUN_ID": "1", "GITHUB_WORKFLOW": "w"})  # no GITHUB_ACTIONS=true
    assert ci["in_ci"] is False and ci["sha"] is None and ci["run_id"] is None
    audit = capability.sha_audit({"head_sha": {"value": SHA}}, ci)
    assert audit["status"] == "local_only" and audit["ci_sha"] is None


def test_ci_facts_come_only_from_the_runner_and_must_equal_the_checked_out_head():
    env = {"GITHUB_ACTIONS": "true", "GITHUB_SHA": SHA, "GITHUB_RUN_ID": "42", "GITHUB_WORKFLOW": "ci", "GITHUB_REPOSITORY": "o/r"}
    ci = capability.probe_ci(env)
    assert ci["in_ci"] and ci["sha"] == SHA and ci["run_id"] == "42"
    assert capability.sha_audit({"head_sha": {"value": SHA}}, ci)["status"] == "verified"
    assert capability.sha_audit({"head_sha": {"value": "b" * 40}}, ci)["status"] == "mismatch"
    assert capability.sha_audit({"head_sha": {"value": None}}, ci)["status"] == "unverifiable"
    assert capability.probe_ci({**env, "GITHUB_SHA": "not-a-sha"})["sha"] is None  # a malformed SHA is not accepted


def test_the_github_cli_probe_keeps_only_the_exit_status(monkeypatch):
    monkeypatch.setattr(capability.shutil, "which", lambda _n: None)
    assert capability.probe_github_cli() == {"gh_installed": False, "authenticated": {"value": None, "reason": "gh executable not found"}}

    class R:
        returncode, stdout, stderr = 0, "Logged in as someone", ""

    monkeypatch.setattr(capability.shutil, "which", lambda _n: "gh")
    monkeypatch.setattr(capability.subprocess, "run", lambda *a, **k: R())
    out = capability.probe_github_cli()
    assert out["authenticated"] == {"value": True, "reason": None} and "someone" not in str(out)


def test_the_probe_command_writes_a_non_blocking_report(tmp_path):
    r = CliRunner().invoke(app, ["probe", "--repo-root", str(tmp_path)])
    assert r.exit_code == 0 and (tmp_path / "outputs" / "capability_probe.json").exists()
