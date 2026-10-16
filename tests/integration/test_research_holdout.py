"""Research-lane holdout opening procedure (spec 11.6, 14): the sealed holdout of the SYNTHETIC reduced research run (INV-HOLD-001, INV-DATA-001).

Same mechanics as the fixture lane (frozen selection and pre-holdout report required, flag + acknowledgement, one audited opening, pre-holdout artifacts
byte-identical afterwards) plus the research bindings: the audit names the snapshot and evidence class, the post-holdout report keeps the synthetic
label, the regime fits the holdout used are persisted, and a synthetic snapshot cannot be opened without the explicit engineering opt-in.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline.checkpoints import LineageMismatch
from quant_pipeline.cli import app
from quant_pipeline.holdout import HoldoutRefused, audit_key, is_open, open_holdout, tree_hash
from quant_pipeline.research_pipeline import EVIDENCE_SYNTHETIC, ResearchPipeline
from support.research_smoke import SMOKE_WINDOW, write_smoke_config
from support.synthetic_research import build_research_repo

ACK = {"QUANT_FINAL_HOLDOUT_ACK": "I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT"}


def attempts(out: Path, run: str = "r1") -> list[str]:
    f = out / run / "holdout_attempts.log"
    return f.read_text(encoding="utf-8").splitlines() if f.exists() else []


def holdout_dirs(out: Path, run: str = "r1") -> list[str]:
    return sorted(x.parent.name for x in (out / run / "checkpoints").rglob("COMPLETE") if "fold_id=holdout" in str(x))


@pytest.fixture(scope="module")
def sealed(research_base, tmp_path_factory):
    """Own copies of the pristine sealed run: one that gets opened, one kept sealed for the negative controls."""
    dest = tmp_path_factory.mktemp("rh")
    root, spare = dest / "work", dest / "spare"
    shutil.copytree(research_base["pristine"], root, symlinks=True)
    shutil.copytree(research_base["pristine"], spare, symlinks=True)
    cfg = research_base["cfg"]
    p = ResearchPipeline(root, root / "out", "r1", config=cfg, allow_synthetic=True)
    return {"root": root, "cfg": cfg, "out": root / "out", "p": p, "proj": research_base["proj"], "spare": spare}


def test_before_opening_the_selection_is_frozen_the_report_exists_and_no_holdout_artifact_exists(sealed):
    p = sealed["p"]
    assert p.store.validate(p.key("11_oos_evaluation", "final", "freeze"), p.lineage)[0]
    assert p.store.validate(p.key("12_bootstrap_and_reports", "global", "pre-holdout"), p.lineage)[0]
    sel = json.loads((p.store.path(p.key("11_oos_evaluation", "final", "freeze")) / "final_selection.json").read_text(encoding="utf-8"))
    assert sel["frozen"] and sel["holdout_sealed"] and sel["created_before_holdout_access"] and "hmm_filtered" in sel["model_family"]
    assert sel["holdout_geometry"]["positions"] == [p.manifest.holdout.start_pos, p.manifest.holdout.end_pos]
    assert {"nav", "returns", "sharpe", "total_return", "fills", "decisions"}.isdisjoint(sel)  # geometry and rules only
    cov = json.loads((p.store.path(p.key("13_verification", "global", "coverage")) / "fixture_coverage.json").read_text(encoding="utf-8"))
    assert cov["holdout"]["opened"] is False and not is_open(p) and holdout_dirs(sealed["out"]) == []


def test_opening_is_refused_without_the_flag_or_the_exact_acknowledgement_and_computes_nothing(sealed):
    p, out = sealed["p"], sealed["out"]
    with pytest.raises(HoldoutRefused) as e1:
        open_holdout(p, unlock=False, environ=ACK)
    assert e1.value.reason == "missing_unlock_flag"
    for env in ({}, {"QUANT_FINAL_HOLDOUT_ACK": "yes"}, {"QUANT_FINAL_HOLDOUT_ACK": ""}):
        with pytest.raises(HoldoutRefused) as e2:
            open_holdout(p, unlock=True, environ=env)
        assert e2.value.reason == "missing_or_wrong_acknowledgement"
    assert len(attempts(out)) == 4 and not is_open(p) and holdout_dirs(out) == []


def test_the_cli_refuses_a_synthetic_snapshot_without_the_engineering_opt_in_and_without_the_flag(sealed):
    base = ["holdout", "open", "--config", str(sealed["spare"] / sealed["cfg"]), "--run-id", "r1", "--repo-root", str(sealed["spare"])]
    r = CliRunner().invoke(app, base, env=ACK)
    assert r.exit_code == 1 and "synthetic" in r.output
    r = CliRunner().invoke(app, [*base, "--allow-synthetic"], env=ACK)  # opt-in given, but not --unlock-final-holdout
    assert r.exit_code == 1 and "missing_unlock_flag" in r.output
    assert holdout_dirs(sealed["spare"] / "out") == []


def test_a_tampered_pre_holdout_artifact_stops_the_opening_before_anything_is_computed(sealed):
    spare = sealed["spare"]
    victim = min((spare / "out" / "r1" / "checkpoints").rglob("strategy.nav.json"))
    victim.write_bytes(victim.read_bytes().replace(b"0", b"1", 1))
    p = ResearchPipeline(spare, spare / "out", "r1", config=sealed["cfg"], allow_synthetic=True)
    with pytest.raises((HoldoutRefused, LineageMismatch, ValueError)):
        open_holdout(p, unlock=True, environ=ACK)
    assert holdout_dirs(spare / "out") == [] and not is_open(p)


def test_a_valid_opening_is_audited_bound_to_the_snapshot_and_leaves_pre_holdout_artifacts_byte_identical(sealed):
    p, out = sealed["p"], sealed["out"]
    pre_report = (p.store.path(p.key("12_bootstrap_and_reports", "global", "pre-holdout")) / "report.md").read_bytes()
    result = open_holdout(p, unlock=True, environ=ACK)
    a = result["audit"]
    assert a["opening_number"] == 1 and a["max_openings_per_lineage"] == 1 and a["pre_holdout_unchanged"] and a["minima"]["met"]
    assert a["pre_holdout_tree_hash_before"] == a["pre_holdout_tree_hash_after"]
    assert a["evidence_class"] == EVIDENCE_SYNTHETIC and a["synthetic_snapshot"] is True and a["snapshot_id"] == p.snapshot.snapshot_id
    assert a["refit_policy"] == "forward_refit_frozen_hyperparameters" and a["final_selection_trial"] == p.outcomes[-1].winner_hash
    assert is_open(p) and p.store.validate(audit_key(p), p.lineage)[0]
    assert (p.store.path(p.key("12_bootstrap_and_reports", "global", "pre-holdout")) / "report.md").read_bytes() == pre_report
    assert tree_hash(out / "r1" / "checkpoints", ("fold_id=holdout", "trial_hash=post-holdout")) == a["pre_holdout_tree_hash_before"]
    assert any("opened: audit" in x for x in attempts(out))
    assert a["minima"]["decisions"] == p.manifest.holdout.n_decisions


def test_the_holdout_used_the_frozen_winner_decides_only_on_scheduled_rows_and_keeps_a_continuous_ledger(sealed):
    p = sealed["p"]
    final = p.outcomes[-1]
    run = p.store.path(p.key("11_oos_evaluation", "holdout-x1", f"selected-{final.winner_hash[7:19]}"))
    assert json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))["meta"]["winner"] == final.winner_hash
    decisions = json.loads((run / "strategy.decisions.json").read_text(encoding="utf-8"))
    assert [d["position"] for d in decisions] == list(p.manifest.holdout.decision_positions)
    assert all(d["position"] in p.manifest.decision_positions for d in decisions)  # a subset of the model rows, never every row
    nav = json.loads((run / "strategy.nav.json").read_text(encoding="utf-8"))
    last_oos = p.outer_runs[1.0][-1].result.nav
    first_scored = p.manifest.holdout.valuation_sessions[0]
    assert min(nav) > max(last_oos)  # the ledger continues after the last outer test: no reset, no overlap
    # OQ-048: the last outer test ends on the row before the holdout, so no session is valued with a carried book without being decided on or scored
    assert p.manifest.outer[-1].test.end_pos + 1 == p.manifest.holdout.start_pos and min(nav) == first_scored
    assert "unscored_carry_sessions" not in json.loads((p.store.path(audit_key(p)) / "holdout_audit.json").read_text(encoding="utf-8"))
    assert all(d["position"] >= p.manifest.holdout.start_pos for d in decisions)


def test_the_post_holdout_report_keeps_the_synthetic_label_and_its_decision_count_matches_the_audit(sealed):
    p = sealed["p"]
    post = p.store.path(p.key("12_bootstrap_and_reports", "global", "post-holdout"))
    text = (post / "report.md").read_text(encoding="utf-8")
    n = p.manifest.holdout.n_decisions
    assert "SYNTHETIC ENGINEERING TEST" in text and "fixture" not in text.lower()
    assert "Final holdout: **opened once (audited)**" in text and f"Holdout ({n} decisions," in text
    assert json.loads((post / "terminology_scan.json").read_text(encoding="utf-8")) == {"forbidden": [], "missing": []}


def test_the_regime_fits_the_holdout_used_are_persisted_as_evidence(sealed):
    p = sealed["p"]
    ev = json.loads((p.store.path(p.key("06_hmm_filtered_state", "holdout")) / "hmm_fits_holdout.json").read_text(encoding="utf-8"))
    run = p.store.path(p.key("11_oos_evaluation", "holdout-x1", f"selected-{p.outcomes[-1].winner_hash[7:19]}"))
    decisions = json.loads((run / "strategy.decisions.json").read_text(encoding="utf-8"))
    assert ev["evidence_class"] == EVIDENCE_SYNTHETIC and ev["winner"] == p.outcomes[-1].winner_hash
    assert ev["refit_positions"] == sorted({d["refit_position"] for d in decisions}) and ev["n_fits"] == len(ev["refit_positions"]) >= 1
    assert all(f["refit_id"] and f["permutation"] is not None for f in ev["fits"])


def test_a_second_opening_is_refused_and_changes_nothing(sealed):
    p, out = sealed["p"], sealed["out"]
    before = tree_hash(out / "r1" / "checkpoints")
    with pytest.raises(HoldoutRefused) as e:
        open_holdout(p, unlock=True, environ=ACK)
    assert e.value.reason == "already_opened" and tree_hash(out / "r1" / "checkpoints") == before
    assert any("already opened" in x for x in attempts(out))


def test_data_observed_after_the_pre_holdout_fence_cannot_change_anything_decided_before_the_opening(sealed, tmp_path):
    """Rewrite every return, FX quote and rate observed after the last pre-holdout session; selection, outer ledgers and the bootstrap must not move
    while the opened holdout does (so the control is not vacuous)."""
    p = sealed["p"]
    last_pre = p.rm.eligible[p.manifest.holdout.start_pos - 2].strftime("%Y-%m-%d")
    root, _ = build_research_repo(tmp_path, start=SMOKE_WINDOW[0], end=SMOKE_WINDOW[1], perturb_after=last_pre)
    cfg = write_smoke_config(root)
    q = ResearchPipeline(root, root / "out", "r1", config=cfg, allow_synthetic=True)
    proj_q = q.run()
    a, b = json.loads(json.dumps(sealed["proj"])), json.loads(json.dumps(proj_q))
    for key in ("winners", "registry", "paths", "bootstrap_index_hash", "oos_intervals"):
        assert a[key] == b[key], key
    assert p.snapshot.snapshot_id != q.snapshot.snapshot_id
    res_q = open_holdout(q, unlock=True, environ=ACK)
    res_p = json.loads((p.store.path(audit_key(p)) / "holdout_audit.json").read_text(encoding="utf-8"))
    assert res_q["audit"]["audit_hash"] != res_p["audit_hash"]
    final = p.outcomes[-1].winner_hash[7:19]
    nav_p = (p.store.path(p.key("11_oos_evaluation", "holdout-x1", f"selected-{final}")) / "strategy.nav.json").read_bytes()
    nav_q = (q.store.path(q.key("11_oos_evaluation", "holdout-x1", f"selected-{final}")) / "strategy.nav.json").read_bytes()
    assert nav_p != nav_q
