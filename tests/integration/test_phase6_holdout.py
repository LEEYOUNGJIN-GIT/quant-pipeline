"""Sealed holdout (spec 11.6, 14, 21 items 3, 7, 23): freeze before access, flag + acknowledgement, one audited opening,
pre-holdout artifacts unchanged, and holdout data unable to influence anything decided before it. Evidence class: fixture_path."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_pipeline.cli import app
from quant_pipeline.holdout import HoldoutRefused, audit_key, is_open, open_holdout, tree_hash
from quant_pipeline.pipeline import FixturePipeline

ROOT = Path(__file__).resolve().parents[2]
ACK = {"QUANT_FINAL_HOLDOUT_ACK": "I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT"}


def lp(p: Path) -> Path:
    prefix = "\\\\?\\"
    s = str(Path(p).resolve())
    return Path(prefix + s) if sys.platform == "win32" and not s.startswith(prefix) else Path(s)


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    out = tmp_path_factory.mktemp("holdout") / "out"
    p = FixturePipeline(ROOT, out, "h1")
    proj = p.run()
    return {"p": p, "out": out, "projection": proj}


def checkpoint_dirs(out: Path, run="h1"):
    return sorted(x.parent.relative_to(lp(out)).as_posix() for x in lp(out / run / "checkpoints").rglob("COMPLETE"))


def attempts(out: Path, run="h1") -> list[str]:
    f = out / run / "holdout_attempts.log"
    return f.read_text(encoding="utf-8").splitlines() if f.exists() else []


def test_before_opening_the_selection_is_frozen_the_report_is_complete_and_nothing_of_the_holdout_exists(sealed):
    p, out = sealed["p"], sealed["out"]
    assert p.store.validate(p.key("11_oos_evaluation", "final", "freeze"), p.lineage)[0]
    rep = p.key("12_bootstrap_and_reports", "global", "pre-holdout")
    assert p.store.validate(rep, p.lineage)[0]
    md = (p.store.path(rep) / "report.md").read_text(encoding="utf-8")
    assert "Final holdout: **sealed (unopened)**" in md and "Tax is a zero placeholder" in md
    assert not is_open(p) and not [d for d in checkpoint_dirs(out) if "fold_id=holdout" in d]
    sel = json.loads((p.store.path(p.key("11_oos_evaluation", "final", "freeze")) / "final_selection.json").read_text(encoding="utf-8"))
    assert sel["frozen"] and sel["created_before_holdout_access"] and sel["refit_policy"] == "forward_refit_frozen_hyperparameters"
    assert sel["trial_hash"] == p.outcomes[-1].winner_hash and sel["holdout_geometry"]["positions"] == [61, 70]
    assert {"nav", "returns", "sharpe", "total_return", "fills", "decisions"}.isdisjoint(sel)  # geometry and rules only: no outcome fields
    assert sel["holdout_geometry"].keys() == {"positions", "valuation_sessions"}


def test_the_pre_holdout_report_passes_the_terminology_scan_and_states_the_required_labels(sealed):
    p = sealed["p"]
    d = p.store.path(p.key("12_bootstrap_and_reports", "global", "pre-holdout"))
    scan = json.loads((d / "terminology_scan.json").read_text(encoding="utf-8"))
    assert scan == {"forbidden": [], "missing": []}
    text = (d / "report.md").read_text(encoding="utf-8")
    for must in ("KRW cash reference", "turnover-matched periodic equal weight", "turnover-matched staged-entry equal-weight hold",
                 "conventional buy-and-hold reference", "initial turnover exempt", "non-inferential", "selection_unstable",
                 "below the fixture minima", "No accepted dependency lock", "Multiple-trial warning"):
        assert must in text, must


def test_opening_is_refused_without_the_flag_or_the_acknowledgement_and_computes_nothing(sealed):
    p, out = sealed["p"], sealed["out"]
    with pytest.raises(HoldoutRefused) as e1:
        open_holdout(p, unlock=False, environ=ACK)
    assert e1.value.reason == "missing_unlock_flag"
    for env in ({}, {"QUANT_FINAL_HOLDOUT_ACK": "yes"}, {"QUANT_FINAL_HOLDOUT_ACK": ""}):
        with pytest.raises(HoldoutRefused) as e2:
            open_holdout(p, unlock=True, environ=env)
        assert e2.value.reason == "missing_or_wrong_acknowledgement"
    assert len(attempts(out)) == 4 and not is_open(p)  # every refusal is logged and none is counted as an opening
    assert not [d for d in checkpoint_dirs(out) if "fold_id=holdout" in d]


def test_holdout_data_cannot_influence_selection_outer_results_or_the_bootstrap(sealed, tmp_path):
    """Perturb every observation that happens after the pre-holdout fence; nothing decided before the opening may move."""
    fence = pd.Timestamp(sealed["p"].manifest.outer[-1].test.fence_utc)
    alt = tmp_path / "alt"
    shutil.copytree(ROOT / "data", alt / "data")
    shutil.copytree(ROOT / "configs", alt / "configs")
    shutil.copy(ROOT / "IMPLEMENTATION_SPEC.md", alt / "IMPLEMENTATION_SPEC.md")
    for name in ("prices_raw", "prices_total_return", "fx_krw_per_usd"):
        f = alt / "data" / "fixtures" / "fixture_3asset" / f"{name}.parquet"
        df = pd.read_parquet(f)
        late = pd.to_datetime(df.available_at_utc, utc=True) > fence
        df.loc[late, "value"] = df.loc[late, "value"] * 1.25
        df.to_parquet(f, index=False)
    proj = tmp_path / "alt_projection.json"
    r = subprocess.run([sys.executable, "-m", "quant_pipeline.pipeline", "--run-id", "alt", "--out-root", str(tmp_path / "alt_out"),
                        "--root", str(alt), "--projection-out", str(proj)], capture_output=True, text=True, cwd=ROOT, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    a, b = json.loads(json.dumps(sealed["projection"])), json.loads(proj.read_text(encoding="utf-8"))  # same JSON normalisation
    for key in ("winners", "registry", "paths", "bootstrap_index_hash", "oos_intervals"):
        assert a[key] == b[key], key  # unchanged: the holdout period had no influence on selection, OOS ledgers or the bootstrap
    assert a["lineage"]["data_manifest_hash"] == b["lineage"]["data_manifest_hash"]  # the manifest hash is of the unperturbed copy


def test_a_valid_opening_is_audited_one_time_and_leaves_pre_holdout_artifacts_byte_identical(sealed):
    p, out = sealed["p"], sealed["out"]
    pre_report = (p.store.path(p.key("12_bootstrap_and_reports", "global", "pre-holdout")) / "report.md").read_bytes()
    result = open_holdout(p, unlock=True, environ=ACK)
    a = result["audit"]
    assert a["opening_number"] == 1 and a["max_openings_per_lineage"] == 1 and a["pre_holdout_unchanged"]
    assert a["pre_holdout_tree_hash_before"] == a["pre_holdout_tree_hash_after"]
    assert a["minima"]["met"] and a["minima"]["decisions"] == 10 and a["minima"]["intervals"] == 12 and a["minima"]["fills"] >= 8
    assert is_open(p) and p.store.validate(audit_key(p), p.lineage)[0]
    assert (p.store.path(p.key("12_bootstrap_and_reports", "global", "pre-holdout")) / "report.md").read_bytes() == pre_report
    assert tree_hash(out / "h1" / "checkpoints", ("fold_id=holdout", "trial_hash=post-holdout")) == a["pre_holdout_tree_hash_before"]
    assert any("opened: audit" in x for x in attempts(out))


def test_the_holdout_used_the_frozen_hyperparameters_forward_refits_and_a_continuous_ledger(sealed):
    p = sealed["p"]
    final = p.outcomes[-1]
    run = p.store.path(p.key("11_oos_evaluation", "holdout-x1", f"selected-{final.winner_hash[7:19]}"))
    meta = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))["meta"]
    assert meta["winner"] == final.winner_hash and meta["multiplier"] == 1.0
    decisions = json.loads((run / "strategy.decisions.json").read_text(encoding="utf-8"))
    assert [d["position"] for d in decisions] == list(range(61, 71)) and {d["refit_position"] for d in decisions} == {61, 66}
    nav = json.loads((run / "strategy.nav.json").read_text(encoding="utf-8"))
    last_oos = p.outer_runs[1.0][-1].result.nav
    assert min(nav) == "2024-05-03" and max(last_oos) == "2024-05-02"  # contiguous: no reset, no overlap with the outer tests
    crossing = [f for f in json.loads((run / "strategy.fills.json").read_text(encoding="utf-8")) if f["decision_position"] == 60]
    assert crossing and all(f["origin"]["fold_id"] == "outer-2" for f in crossing)  # pre-holdout orders cross the boundary once


def test_three_cost_ledgers_exist_for_the_holdout_and_it_never_enters_the_bootstrap(sealed):
    p = sealed["p"]
    final = p.outcomes[-1]
    costs = {}
    for m in (0.0, 1.0, 2.0):
        run = p.store.path(p.key("11_oos_evaluation", f"holdout-x{m:g}", f"selected-{final.winner_hash[7:19]}"))
        costs[m] = json.loads((run / "strategy.meta.json").read_text(encoding="utf-8"))["costs"]
    assert float(costs[0.0]) == 0.0 and float(costs[1.0]) > 0 and float(costs[2.0]) > 0
    assert p.summary["oos_intervals"]["strategy"] == 22 and p.summary["bootstrap_shape"] == [200, 22]  # holdout excluded


def test_a_second_opening_is_refused_and_changes_nothing(sealed):
    p, out = sealed["p"], sealed["out"]
    before = tree_hash(out / "h1" / "checkpoints")
    with pytest.raises(HoldoutRefused) as e:
        open_holdout(p, unlock=True, environ=ACK)
    assert e.value.reason == "already_opened" and tree_hash(out / "h1" / "checkpoints") == before
    assert any("already opened" in x for x in attempts(out))


def test_the_post_holdout_report_is_a_new_artifact_with_the_holdout_in_its_own_section(sealed):
    p = sealed["p"]
    post = p.store.path(p.key("12_bootstrap_and_reports", "global", "post-holdout"))
    text = (post / "report.md").read_text(encoding="utf-8")
    assert "Final holdout: **opened once (audited)**" in text and "not part of the bootstrap or any selection" in text
    assert json.loads((post / "terminology_scan.json").read_text(encoding="utf-8")) == {"forbidden": [], "missing": []}


def test_the_cli_refuses_to_open_without_the_flag(sealed):
    cfg = ROOT / "configs" / "fixture_3asset.yaml"
    r = CliRunner().invoke(app, ["holdout", "open", "--config", str(cfg), "--run-id", "h1", "--repo-root", str(ROOT)])
    assert r.exit_code == 1 and "holdout NOT opened" in (r.output + str(r.stderr_bytes if hasattr(r, "stderr_bytes") else ""))


def test_pairwise_agent_diagnostics_exist_per_scope_and_the_holdout_scope_appears_only_after_opening(sealed):
    p = sealed["p"]
    pre = json.loads((p.store.path(p.key("13_verification", "global", "agent-diagnostics")) / "agent_pairwise_diagnostics.json").read_text(encoding="utf-8"))
    assert set(pre["scopes"]) == {"calibration", "outer_oos"} and pre["identity_threshold"] is None
    assert pre["scopes"]["outer_oos"]["valid_simultaneous_decisions"] == 20 and len(pre["scopes"]["calibration"]["pairs"]) == 6
    pair = pre["scopes"]["outer_oos"]["pairs"]["Defensive|Diversification"]["risky_share_cosine"]
    assert pair["count"] > 0 and 0.5 < pair["median"] <= 1.0  # measured, reported, not thresholded (OQ-018)
    post = json.loads((p.store.path(p.key("13_verification", "holdout", "agent-diagnostics")) / "agent_pairwise_diagnostics.json").read_text(encoding="utf-8"))
    assert set(post["scopes"]) == {"holdout"} and post["scopes"]["holdout"]["valid_simultaneous_decisions"] == 10


def test_the_coverage_matrix_counts_what_the_run_exercised_and_reports_no_uncovered_row(sealed):
    p = sealed["p"]
    m = json.loads((p.store.path(p.key("13_verification", "global", "coverage")) / "fixture_coverage.json").read_text(encoding="utf-8"))
    r = m["rows"]
    assert m["uncovered"] == [] and r["inner_validation"]["trial_fold_evaluations"] == 48 and r["inner_validation"]["blocks"] == 6
    assert r["outer_oos"] == {"folds": 2, "decisions": 20, "fills": r["outer_oos"]["fills"], "scored_intervals": 22} and r["outer_oos"]["fills"] > 0
    assert set(r["cost_scenarios"]) == {"0x", "1x", "2x"} and float(r["cost_scenarios"]["0x"]["realized_costs"]) == 0.0
    assert float(r["cost_scenarios"]["2x"]["realized_costs"]) > float(r["cost_scenarios"]["1x"]["realized_costs"]) > 0
    assert r["bootstrap"]["shape"] == [200, 22] and len(r["benchmarks"]) == 4 and r["refit"]["scheduled_positions"] >= r["refit"]["cores_used_by_decisions"] > 1
    assert r["checkpoint_stages"]["10_event_time_execution_and_accounting"] == 8 and r["holdout"]["geometry_decisions"] == 10
    assert m["generation_lock"]["scenario_id"] == "fixture-v5"
