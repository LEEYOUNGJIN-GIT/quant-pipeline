"""Graph-derived compute plan and the after-plan gate (spec 11.8, 13, 18). No fixed historical total is asserted: every number is
recomputed from the same graph the plan was built from."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quant_pipeline.calendars import Calendars, build_row_map
from quant_pipeline.cli import app
from quant_pipeline.config import Config, load_config, load_yaml_strict
from quant_pipeline.fitstate import lineage_hashes
from quant_pipeline.plan import build_compute_plan, doctor_after_plan, write_plan
from quant_pipeline.splits import build_split_manifest
from quant_pipeline.trials import enumerate_trials

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
CALS = Calendars.load(["XKRX", "XNYS"])
RM, _ = build_row_map(CFG, CALS)
HASHES = lineage_hashes(CFG, ROOT)


def make(cfg=CFG):
    return build_compute_plan(cfg, build_split_manifest(cfg, CALS, RM), enumerate_trials(cfg), HASHES)


PLAN = make()
AMPLE = {"logical_cpu": 8.0, "available_ram_gib": 64.0, "free_disk_gib": 500.0}


@pytest.fixture(autouse=True)
def ample_host(monkeypatch):
    """Host capacity is machine state; the tests pin it so the graph and ceiling logic are what is tested."""
    monkeypatch.setattr("quant_pipeline.plan.host_capacity", lambda: dict(AMPLE))


def test_insufficient_host_ram_blocks_the_gate(monkeypatch, tmp_path):
    monkeypatch.setattr("quant_pipeline.plan.host_capacity", lambda: {**AMPLE, "available_ram_gib": 1.0})
    assert "host_ram" in report(CFG, PLAN, tmp_path, "low-ram")["failed"]


def test_every_count_is_recomputed_from_the_execution_graph():
    m, d = build_split_manifest(CFG, CALS, RM), enumerate_trials(CFG)
    g = PLAN["graph"]
    committed = sum(e - s + 1 for s, e in [(o.test.start_pos, o.test.end_pos) for o in m.outer] + [(m.holdout.start_pos, m.holdout.end_pos)])
    mults = len(CFG.costs.sensitivity_multipliers)
    last = max(m.path_endpoints())
    assert g["committed_decisions"] == committed and g["trials_planned"] == len(d.trials) == 8
    assert PLAN["counts"]["allocation_optimizer_solve"]["deduplicated"] == len(d.trials) * last + committed * mults
    assert PLAN["counts"]["allocation_optimizer_solve"]["no_sharing_upper_bound"] == len(d.trials) * sum(m.path_endpoints()) + committed * mults
    assert PLAN["counts"]["model_fit_unit"]["deduplicated"] == len(m.refit_positions) + 1  # distinct refits + the HMM probe
    assert PLAN["counts"]["bootstrap_replicate"]["deduplicated"] == CFG.bootstrap.replicates


def test_counts_scale_with_the_graph_not_with_constants():
    fewer = CFG.model_copy(update={"evaluation": CFG.evaluation.model_copy(update={"max_parameter_trials": 4})})
    p4 = make(fewer)
    assert p4["graph"]["trials_planned"] == 4
    assert p4["counts"]["allocation_optimizer_solve"]["deduplicated"] < PLAN["counts"]["allocation_optimizer_solve"]["deduplicated"]
    assert p4["counts"]["allocation_optimizer_solve"]["deduplicated"] - 30 * 3 == 4 * 60  # committed part unchanged, inner part is 4 x 60
    two = CFG.model_copy(update={"costs": CFG.costs.model_copy(update={"sensitivity_multipliers": [0.0, 1.0]})})
    assert make(two)["counts"]["allocation_optimizer_solve"]["deduplicated"] == 8 * 60 + 30 * 2


def test_deduplicated_never_exceeds_the_no_sharing_upper_bound():
    for unit, c in PLAN["counts"].items():
        assert c["deduplicated"] <= c["no_sharing_upper_bound"], unit
    t = PLAN["total_optimizer_solves"]
    assert t["deduplicated_solver_attempts_upper_bound"] == t["deduplicated_problems"] * 2 and t["max_attempts_per_problem"] == 2
    assert t["no_sharing_solver_attempts_upper_bound"] > t["cap"]  # the conservative bound is reported beside the planned count


def test_the_plan_is_hashed_stable_and_sensitive_to_the_graph():
    assert make()["plan_hash"] == PLAN["plan_hash"] and make()["dependency_graph_hash"] == PLAN["dependency_graph_hash"]
    other = make(CFG.model_copy(update={"evaluation": CFG.evaluation.model_copy(update={"max_parameter_trials": 4})}))
    assert other["plan_hash"] != PLAN["plan_hash"] and other["dependency_graph_hash"] != PLAN["dependency_graph_hash"]
    assert PLAN["operation_caps_from_plan"] is False and PLAN["graph"]["split_manifest_hash"].startswith("sha256:")


def test_the_plan_file_is_immutable(tmp_path):
    p = write_plan(tmp_path, "r", PLAN)
    assert json.loads(p.read_text(encoding="utf-8"))["plan_hash"] == PLAN["plan_hash"]
    with pytest.raises(FileExistsError):
        write_plan(tmp_path, "r", PLAN)


def report(cfg, plan, tmp_path, run_id="r"):
    (tmp_path / run_id).mkdir(exist_ok=True)
    return doctor_after_plan(cfg, plan, tmp_path / run_id / "compute_plan.json", tmp_path, run_id)


def test_the_fixture_plan_exceeds_one_profile_ceiling_and_the_gate_reports_it(tmp_path):
    r = report(CFG, PLAN, tmp_path)
    assert r["status"] == "fail" and r["failed"] == ["within_cap[auxiliary_optimizer_solve]"]
    c = next(x for x in r["checks"] if x["check"] == "within_cap[auxiliary_optimizer_solve]")
    assert "deduplicated 525 vs cap 512" in c["detail"] or ("vs cap 512" in c["detail"] and "deduplicated" in c["detail"])
    assert PLAN["counts"]["auxiliary_optimizer_solve"]["deduplicated"] > CFG.compute_budget.max_auxiliary_optimizer_solves
    assert (tmp_path / "r" / "doctor.after_plan.json").exists() and r["plan_hash"] == PLAN["plan_hash"]
    assert r["note"].startswith("caps are independent ceilings")


def test_every_other_ceiling_and_the_host_checks_pass(tmp_path):
    r = report(CFG, PLAN, tmp_path)
    bad = [c["check"] for c in r["checks"] if c["status"] == "fail"]
    assert bad == ["within_cap[auxiliary_optimizer_solve]"]
    names = {c["check"] for c in r["checks"]}
    assert {"host_cpu", "host_ram", "host_disk", "wall_estimate_within_budget", "plan_hash_matches_content",
            "dependency_graph_hash_matches", "candidate_cardinality_within_cap"} <= names


def test_a_formally_revised_ceiling_lets_the_gate_pass_but_only_through_a_recorded_change(tmp_path):
    revised = CFG.model_copy(update={"compute_budget": CFG.compute_budget.model_copy(update={"max_auxiliary_optimizer_solves": 1024})})
    replanned = make(revised)  # a revised profile requires a new plan, whose recorded ceiling matches the profile
    assert replanned["plan_hash"] != PLAN["plan_hash"] and report(revised, replanned, tmp_path)["status"] == "pass"


def test_a_cap_copied_from_the_plan_or_unbounded_is_invalid(tmp_path):
    planned = PLAN["counts"]["allocation_optimizer_solve"]["deduplicated"]
    copied = CFG.model_copy(update={"compute_budget": CFG.compute_budget.model_copy(update={"max_allocation_optimizer_solves": planned})})
    r = report(copied, PLAN, tmp_path)
    assert "cap_is_independent[allocation_optimizer_solve]" in r["failed"]
    zero = CFG.model_copy(update={"compute_budget": CFG.compute_budget.model_copy(update={"max_checkpoints": 0})})
    assert "cap_is_independent[checkpoint]" in report(zero, PLAN, tmp_path, "r2")["failed"]
    assert "cap_is_independent[bootstrap_replicate]" not in r["failed"]  # equal to the configured replicate count by definition


def test_a_tampered_plan_is_rejected(tmp_path):
    forged = copy.deepcopy(PLAN)
    forged["counts"]["allocation_optimizer_solve"]["deduplicated"] = 1
    assert "plan_hash_matches_content" in report(CFG, forged, tmp_path)["failed"]
    forged2 = copy.deepcopy(PLAN)
    forged2["operation_caps_from_plan"] = True
    assert "caps_not_copied_from_plan" in report(CFG, forged2, tmp_path, "r2")["failed"]


# ----------------------------------------------------------------------------------------------- CLI gate
runner = CliRunner()


def repo(tmp_path, aux_cap=None) -> Path:
    root = tmp_path / "repo"
    for name in ("configs", "data"):
        shutil.copytree(ROOT / name, root / name)
    shutil.copy(ROOT / "IMPLEMENTATION_SPEC.md", root / "IMPLEMENTATION_SPEC.md")
    if aux_cap:
        f = root / "configs" / "fixture_3asset.yaml"
        raw = load_yaml_strict(f.read_text(encoding="utf-8"))
        raw["compute_budget"]["max_auxiliary_optimizer_solves"] = aux_cap
        Config.model_validate(raw)
        f.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return root


def cli(root, *args):
    return runner.invoke(app, [*args, "--config", str(root / "configs" / "fixture_3asset.yaml"), "--repo-root", str(root)])


def test_run_refuses_without_a_plan_and_without_a_passing_gate(tmp_path):
    root = repo(tmp_path)
    assert cli(root, "run", "--run-id", "x").exit_code == 2
    assert cli(root, "plan", "--run-id", "x").exit_code == 0
    assert cli(root, "run", "--run-id", "x").exit_code == 2  # plan exists but doctor --after-plan has not run
    d = cli(root, "doctor", "--after-plan", "--run-id", "x")
    assert d.exit_code == 1 and "auxiliary_optimizer_solve" in d.output
    r = cli(root, "run", "--run-id", "x")
    assert r.exit_code == 1 and "after-plan gate failed" in r.output
    assert not (root / "outputs" / "runs" / "x" / "checkpoints").exists()  # nothing was computed


def test_a_changed_plan_invalidates_the_after_plan_report(tmp_path):
    root = repo(tmp_path, aux_cap=1024)
    assert cli(root, "plan", "--run-id", "y").exit_code == 0 and cli(root, "doctor", "--after-plan", "--run-id", "y").exit_code == 0
    pf = root / "outputs" / "runs" / "y" / "compute_plan.json"
    plan = json.loads(pf.read_text(encoding="utf-8"))
    plan["plan_hash"] = "sha256:" + "0" * 64
    pf.write_text(json.dumps(plan), encoding="utf-8")
    r = cli(root, "run", "--run-id", "y")
    assert r.exit_code == 2 and "different plan" in r.output


def test_with_a_recorded_revision_the_gated_run_completes_end_to_end(tmp_path):
    root = repo(tmp_path, aux_cap=1024)
    assert cli(root, "plan", "--run-id", "z").exit_code == 0
    assert cli(root, "doctor", "--after-plan", "--run-id", "z").exit_code == 0
    out = subprocess.run([sys.executable, "-m", "quant_pipeline.cli", "run", "--config", str(root / "configs" / "fixture_3asset.yaml"),
                          "--run-id", "z", "--repo-root", str(root)], capture_output=True, text=True, cwd=ROOT, check=False)
    assert out.returncode == 0 and "run complete" in out.stdout, out.stdout + out.stderr
    assert (root / "outputs" / "runs" / "z" / "progress_log.txt").exists()


def test_ceilings_come_from_the_current_profile_not_from_the_plan_file(tmp_path):
    forged = copy.deepcopy(PLAN)
    forged["counts"]["auxiliary_optimizer_solve"]["cap"] = 10_000  # a forged, generous cap recorded in the plan
    forged["plan_hash"] = __import__("quant_pipeline.hashing", fromlist=["x"]).hash_json_obj({k: v for k, v in forged.items() if k != "plan_hash"})
    r = report(CFG, forged, tmp_path)
    assert "plan_records_profile_cap[auxiliary_optimizer_solve]" in r["failed"]  # the plan disagrees with the declared profile
    assert "within_cap[auxiliary_optimizer_solve]" in r["failed"]  # and the real ceiling (512) is still the one applied
    revised = CFG.model_copy(update={"compute_budget": CFG.compute_budget.model_copy(update={"max_auxiliary_optimizer_solves": 1024})})
    stale = report(revised, PLAN, tmp_path, "r3")  # the plan was built when the cap was 512
    assert stale["failed"] == ["plan_records_profile_cap[auxiliary_optimizer_solve]"]  # a stale plan is flagged, not silently accepted
