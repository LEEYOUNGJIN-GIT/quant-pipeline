"""The research profile resolves a verified synthetic snapshot, validates its data, builds the row map and plans (INV-DATA-001, offline).

The data are SYNTHETIC (tests/support/synthetic_research.py): they exercise the engine's research path and are not research evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_pipeline.cli import app
from quant_pipeline.config import load_config
from quant_pipeline.data_quality import validate_data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from support.synthetic_research import build_research_repo

CONFIG = str(ROOT / "configs" / "research_8_12asset.yaml")
CFG = load_config(CONFIG)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root, sid = build_research_repo(tmp_path_factory.mktemp("research"))
    return root, sid


def cli(root, *args):
    return CliRunner().invoke(app, [*args, "--config", CONFIG, "--repo-root", str(root)])


def test_the_synthetic_research_snapshot_passes_every_data_check(repo):
    root, _ = repo
    results = validate_data(CFG, root)
    assert [r for r in results if r["status"] != "pass"] == []
    names = {r["check"] for r in results}
    assert {"snapshot_resolved", "row_map", "complete_case_coverage", "ca_reconciliation_level2_total_return", "initial_holdings"} <= names


def test_data_validate_reports_the_row_map_geometry(repo):
    root, _ = repo
    r = cli(root, "data", "validate")
    assert r.exit_code == 0 and "84 monthly decisions" in r.output and "1650 eligible rows" in r.output, r.output


def test_plan_counts_decisions_not_model_rows_and_stays_inside_the_caps(repo):
    root, _ = repo
    r = cli(root, "plan", "--run-id", "rp")
    assert r.exit_code == 0, r.output
    plan = json.loads((root / "outputs" / "runs" / "rp" / "compute_plan.json").read_text(encoding="utf-8"))
    g = plan["graph"]
    assert g["eligible_positions"] == 1650 and g["decision_rows"] == 84  # monthly decisions are a subset of the model rows
    assert g["committed_decisions"] < 84 and plan["profile"] == "research_8_12asset"
    for unit, c in plan["counts"].items():
        assert c["deduplicated"] <= c["cap"], unit
    assert plan["counts"]["rolling_kernel_evaluation"]["deduplicated"] == 84  # one evaluation per decision cutoff, not per model row
    assert plan["counts"]["bootstrap_replicate"]["deduplicated"] == 2000


def test_the_after_plan_gate_fails_only_on_this_machine_not_on_the_plan(repo):
    root, _ = repo
    assert cli(root, "plan", "--run-id", "rp2").exit_code == 0
    r = cli(root, "doctor", "--after-plan", "--run-id", "rp2")
    failed = {line.split(":")[0].split()[-1] for line in r.output.splitlines() if line.startswith("FAIL")}
    assert failed <= {"host_cpu", "host_ram", "host_disk"}, r.output  # the research machine requirements, not plan defects


def test_a_second_plan_for_the_same_run_id_is_refused_as_immutable(repo):
    root, _ = repo
    assert cli(root, "plan", "--run-id", "once").exit_code == 0
    assert cli(root, "plan", "--run-id", "once").exit_code != 0


def test_plan_refuses_a_snapshot_that_does_not_verify(tmp_path):
    root, sid = build_research_repo(tmp_path)
    rates = root / "data" / "snapshots" / "research" / sid / "rates_krw_rfr.parquet"
    rates.write_bytes(rates.read_bytes() + b"x")
    r = cli(root, "plan", "--run-id", "bad")
    assert r.exit_code == 2 and "plan refused" in r.output
    assert not (root / "outputs" / "runs" / "bad").exists()


def test_a_missing_observation_on_a_complete_case_session_halts_the_data_check(tmp_path):
    def drop(tables):
        tr = tables["prices_total_return"]
        victim = tr[(tr.instrument_id == "R01")].exchange_session.iloc[400]
        tables["prices_total_return"] = tr[~((tr.instrument_id == "R01") & (tr.exchange_session == victim))]

    root, _ = build_research_repo(tmp_path, mutate=drop)
    failed = {r["check"]: r["detail"] for r in validate_data(CFG, root) if r["status"] != "pass"}
    assert "complete_case_coverage" in failed and "R01" in failed["complete_case_coverage"]


def test_a_window_too_short_for_one_outer_fold_is_reported_not_truncated(tmp_path):
    root, _ = build_research_repo(tmp_path, start="2018-01-02", end="2019-12-31")
    failed = {r["check"]: r["detail"] for r in validate_data(CFG, root) if r["status"] != "pass"}
    assert "row_map" in failed and "warmup" in failed["row_map"] or "training rows" in failed["row_map"]
    assert cli(root, "plan", "--run-id", "short").exit_code == 2
