from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_pipeline.cli import app

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture
def repo(tmp_path):
    """A minimal copy of the repository inputs so negative controls never touch the real fixture."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "data" / "fixtures", tmp_path / "data" / "fixtures")
    shutil.copy(ROOT / "IMPLEMENTATION_SPEC.md", tmp_path / "IMPLEMENTATION_SPEC.md")
    return tmp_path


def doctor(repo: Path, profile="fixture_3asset"):
    return runner.invoke(app, ["doctor", "--config", str(repo / "configs" / f"{profile}.yaml"), "--repo-root", str(repo)])


def test_static_doctor_passes_and_never_approves_budget(repo):
    r = doctor(repo)
    assert r.exit_code == 0, r.output
    report = json.loads((repo / "outputs" / "doctor.static.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass" and report["approves_compute_budget"] is False
    names = {c["check"] for c in report["checks"]}
    assert {"config_schema", "imports", "calendars", "data_files", "fixture_manifest_hashes", "spec_hash"} <= names


def test_doctor_reports_python_and_lock_deviations_as_nonblocking(repo):
    doctor(repo)
    report = json.loads((repo / "outputs" / "doctor.static.json").read_text(encoding="utf-8"))
    lock = next(c for c in report["checks"] if c["check"] == "accepted_lock")
    assert lock["blocking"] is False


def test_doctor_fails_on_tampered_fixture_file(repo):
    target = repo / "data" / "fixtures" / "fixture_3asset" / "prices_raw.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    r = doctor(repo)
    assert r.exit_code == 1 and "fixture_manifest_hashes" in r.output


def test_doctor_fails_on_missing_fixture_file(repo):
    (repo / "data" / "fixtures" / "fixture_3asset" / "rates_krw_rfr.parquet").unlink()
    r = doctor(repo)
    assert r.exit_code == 1 and "missing" in r.output


def test_doctor_fails_on_invalid_config(repo):
    p = repo / "configs" / "fixture_3asset.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("max_krx_stale_sessions: 2", "max_krx_stale_sessions: 6"), encoding="utf-8")
    r = doctor(repo)
    assert r.exit_code == 1 and "config_schema" in r.output


def test_doctor_after_plan_is_not_faked(repo):
    r = runner.invoke(app, ["doctor", "--config", str(repo / "configs" / "fixture_3asset.yaml"), "--after-plan",
                            "--run-id", "x", "--repo-root", str(repo)])
    assert r.exit_code == 2


def test_config_validate_hash_is_stable_and_invalid_exits_1(repo):
    a = runner.invoke(app, ["config", "validate", "--config", str(ROOT / "configs" / "fixture_3asset.yaml")])
    b = runner.invoke(app, ["config", "validate", "--config", str(ROOT / "configs" / "fixture_3asset.yaml")])
    assert a.exit_code == 0 and a.output == b.output and json.loads(a.output)["resolved_config_hash"].startswith("sha256:")
    bad = repo / "bad.yaml"
    bad.write_text("schema_version: '5.3.2'\nschema_version: '5.3.2'\n", encoding="utf-8")
    assert runner.invoke(app, ["config", "validate", "--config", str(bad)]).exit_code == 1


def test_run_without_a_plan_refuses_honestly(repo):
    r = runner.invoke(app, ["run", "--config", str(repo / "configs" / "fixture_3asset.yaml"), "--repo-root", str(repo)])
    assert r.exit_code == 2 and "refusing to run" in r.output


def test_data_validate_passes_on_the_fixture_and_catches_tampering(repo):
    ok = runner.invoke(app, ["data", "validate", "--config", str(repo / "configs" / "fixture_3asset.yaml"), "--repo-root", str(repo)])
    assert ok.exit_code == 0 and "data validate: pass" in ok.output, ok.output
    import pandas as pd
    p = repo / "data" / "fixtures" / "fixture_3asset" / "prices_raw.parquet"
    df = pd.read_parquet(p)
    df.loc[3, "value"] = -1.0  # a non-positive price
    df.to_parquet(p, index=False)
    bad = runner.invoke(app, ["data", "validate", "--config", str(repo / "configs" / "fixture_3asset.yaml"), "--repo-root", str(repo)])
    assert bad.exit_code == 1 and "FAIL" in bad.output and "manifest_integrity" in bad.output


def test_data_validate_detects_a_row_off_the_official_close(repo):
    import pandas as pd
    p = repo / "data" / "fixtures" / "fixture_3asset" / "prices_raw.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "event_timestamp_utc"] = "2023-12-28T06:00:00Z"
    df.to_parquet(p, index=False)
    r = runner.invoke(app, ["data", "validate", "--config", str(repo / "configs" / "fixture_3asset.yaml"), "--repo-root", str(repo)])
    assert r.exit_code == 1 and "sessions_match_official_calendar" in r.output
