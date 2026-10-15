"""Reduced end-to-end research run on the SYNTHETIC snapshot (INV-DATA-001): engineering evidence, never a research result.

The configuration is the research profile shrunk by tests/support/research_smoke.py (same structure: snapshot input, monthly decisions that are a
subset of the model rows, the HMM-filtered lane, a PCA/gate trial space, a tail-fraction holdout, nested folds, cost scenarios, benchmarks).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline.research_pipeline import EVIDENCE_SYNTHETIC, ResearchPipeline, SyntheticSnapshotRefused
from support.research_smoke import SMOKE_WINDOW, write_smoke_config
from support.synthetic_research import build_research_repo


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root, _ = build_research_repo(tmp_path_factory.mktemp("rp"), start=SMOKE_WINDOW[0], end=SMOKE_WINDOW[1])
    cfg = write_smoke_config(root)
    out = root / "out"
    p = ResearchPipeline(root, out, "r1", config=cfg, allow_synthetic=True)
    proj = p.run()
    return {"root": root, "cfg": cfg, "out": out, "p": p, "proj": proj}


def art(run, stage_dir_part: str, name: str) -> bytes:
    hits = sorted((run["out"] / "r1" / "checkpoints").rglob(name))
    hits = [h for h in hits if stage_dir_part in str(h)]
    assert hits, f"{name} not found under {stage_dir_part}"
    return hits[0].read_bytes()


def test_a_synthetic_snapshot_is_refused_unless_explicitly_allowed(tmp_path):
    root, _ = build_research_repo(tmp_path, start=SMOKE_WINDOW[0], end=SMOKE_WINDOW[1])
    cfg = write_smoke_config(root)
    with pytest.raises(SyntheticSnapshotRefused, match="synthetic"):
        ResearchPipeline(root, root / "out", "no", config=cfg)


def test_the_run_completes_with_decisions_only_at_scheduled_rows(run):
    p = run["p"]
    m = p.manifest
    assert len(m.decision_positions) < m.n_positions and len(m.outer) == 2 and len(p.ev.design.trials) == 4
    for mult, runs in p.outer_runs.items():
        for r, fold in zip(runs, m.outer, strict=True):
            assert [d.position for d in r.result.decisions] == list(fold.test.decision_positions), (mult, fold.fold_id)
            assert all(d.position in m.decision_positions for d in r.result.decisions)


def test_the_trial_space_and_winner_use_the_research_dimensions(run):
    p = run["p"]
    assert p.ev.design.cardinality == 5832 and run["proj"]["winners"].keys() == {"outer-1", "outer-2", "final"}
    for o in p.outcomes:
        assert "pca.n_components_candidates" in o.winner_parameters and "regime.agent_gate_candidates" in o.winner_parameters


def test_every_run_artifact_is_labelled_synthetic_engineering_test(run):
    assert run["proj"]["evidence_class"] == EVIDENCE_SYNTHETIC
    assert run["p"].summary["evidence_class"] == EVIDENCE_SYNTHETIC
    hmm = json.loads(art(run, "06_hmm_filtered_state", "hmm_fits.json"))
    assert hmm["evidence_class"] == EVIDENCE_SYNTHETIC
    ctx = json.loads(art(run, "12_bootstrap_and_reports", "report_context.json"))
    assert "SYNTHETIC ENGINEERING TEST" in ctx["status_banner"] and "research evidence" in ctx["status_banner"]
    md = art(run, "12_bootstrap_and_reports", "report.md").decode("utf-8")
    assert "SYNTHETIC ENGINEERING TEST" in md and "monthly" in md and "fixture" not in md.lower()


def test_the_hmm_fits_are_persisted_and_converged_for_the_frozen_winners(run):
    hmm = json.loads(art(run, "06_hmm_filtered_state", "hmm_fits.json"))
    m = run["p"].manifest
    last = m.outer[-1].test.end_pos
    refits = [r for r in m.refit_positions if r <= last]
    assert hmm["all_converged"] and hmm["seed_stream"] == "hmm_em" and hmm["n_fits"] == len(refits) * len(hmm["winner_pairs"])
    assert hmm["scored_decisions"] == sum(o.test.n_decisions for o in m.outer)
    for f in hmm["fits"]:
        assert f["n_states"] in (2, 3) and f["labels"][-1] == "high_risk" and abs(sum(f["boundary_posterior"]) - 1) < 1e-9
        assert f["fitter"].startswith("hmmlearn==") and f["n_components"] in (1, 2, 3)
    assert {tuple(p) for p in hmm["winner_pairs"]} == {(f["n_components"], f["n_states"]) for f in hmm["fits"]}


def test_the_holdout_stays_sealed(run):
    cov = json.loads(art(run, "13_verification", "fixture_coverage.json"))
    assert cov["holdout"]["opened"] is False and cov["evidence_class"] == EVIDENCE_SYNTHETIC
    fin = json.loads(art(run, "11_oos_evaluation", "final_selection.json"))
    assert fin["holdout_sealed"] is True and fin["created_before_holdout_access"] is True and "hmm_filtered" in fin["model_family"]
    assert not list((run["out"] / "r1").rglob("*holdout_audit*"))  # no opening record exists
    md = art(run, "12_bootstrap_and_reports", "report.md").decode("utf-8")
    assert "sealed" in md.lower()


def test_costs_scenarios_benchmarks_and_the_bootstrap_ran(run):
    p = run["p"]
    assert set(p.outer_runs) == {0.0, 1.0, 2.0}
    r = p.outer_runs[1.0][0]
    assert set(r.benchmarks) >= {"cash_krw", "equal_weight_periodic", "equal_weight_staged_entry", "equal_weight_buy_and_hold_reference"}
    assert p.summary["bootstrap_status"] in ("ok", "complete") or p.summary["bootstrap_shape"][0] == 100


def test_a_second_run_with_the_same_id_resumes_every_checkpoint_and_reproduces_the_projection(run):
    q = ResearchPipeline(run["root"], run["out"], "r1", config=run["cfg"], allow_synthetic=True)
    proj = q.run()
    assert q.computed == [] and len(q.loaded) > 10
    assert proj == run["proj"]


# ---------------------------------------------------------------------------------------------------- interrupted and resumed lineage
def _cli(run, out, *extra):
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "0"}
    return subprocess.run([sys.executable, "-m", "quant_pipeline.research_pipeline", "--root", str(run["root"]), "--out-root", str(out), "--run-id", "r1",
                           "--config", run["cfg"], "--allow-synthetic", *extra], cwd=run["root"], env=env, capture_output=True, text=True, check=False,
                          timeout=3000)


def _artifacts(out: Path) -> dict[str, str]:
    import hashlib

    res = {}
    for f in sorted((out / "r1" / "checkpoints").rglob("*")):
        if f.is_file() and f.name not in ("checkpoint.json", "COMPLETE"):
            res[str(f.relative_to(out / "r1"))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return res


@pytest.fixture(scope="module")
def resumed(run):
    out2 = run["root"] / "out_resume"
    stopped = _cli(run, out2, "--stop-after", "6")
    proj_path = run["root"] / "resumed_projection.json"
    done = _cli(run, out2, "--projection-out", str(proj_path))
    return {"out": out2, "stopped": stopped, "done": done, "proj": json.loads(proj_path.read_text(encoding="utf-8")) if proj_path.exists() else None}


def test_an_interrupted_run_stops_after_a_durable_checkpoint_and_a_new_process_resumes_it(resumed):
    assert resumed["stopped"].returncode == 75 and "STOPPED" in resumed["stopped"].stdout, resumed["stopped"].stdout + resumed["stopped"].stderr
    assert resumed["done"].returncode == 0, resumed["done"].stdout + resumed["done"].stderr
    assert "loaded=6" in resumed["done"].stdout or int(resumed["done"].stdout.split("loaded=")[1].split()[0]) >= 6  # the six written before the stop


def test_the_resumed_lineage_has_the_same_projection_as_the_clean_one(run, resumed):
    assert resumed["proj"] == json.loads(json.dumps(run["proj"], sort_keys=True))


def test_the_resumed_lineage_has_byte_identical_artifacts_in_every_checkpoint(run, resumed):
    clean, again = _artifacts(run["out"]), _artifacts(resumed["out"])
    assert clean.keys() == again.keys() and len(clean) > 30
    assert {k for k in clean if clean[k] != again[k]} == set()
