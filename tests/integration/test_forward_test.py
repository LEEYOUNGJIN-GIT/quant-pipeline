"""Forward test of a frozen selection on a LONGER SYNTHETIC snapshot (tools/forward_test.py): engineering evidence, never a research result.

The reduced research profile is used on a three-year synthetic window; the frozen selection is the first trial of the design, standing in for a finished
run's winner. What is checked: no future data enters earlier results, the frozen selection (not the data) decides the path, the benchmarks start under
the same conditions, and the printed summary passes the wording rules.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location("forward_test_tool", ROOT / "tools" / "forward_test.py")
F = importlib.util.module_from_spec(spec)
sys.modules["forward_test_tool"] = F
spec.loader.exec_module(F)
from support.research_smoke import write_smoke_config
from support.synthetic_research import build_research_repo

WINDOW = ("2020-01-02", "2022-12-30")
START = "2022-01-03"
PERTURB_AFTER = "2022-06-30"


def build(tmp_path_factory, name, perturb_after=None):
    root, _ = build_research_repo(tmp_path_factory.mktemp(name), start=WINDOW[0], end=WINDOW[1], perturb_after=perturb_after)
    cfg = write_smoke_config(root)
    return F.ForwardPipeline(root, root / "out", "f1", config=cfg, allow_synthetic=True)


def frozen(p, i):
    t = p.ev.design.trials[i]
    return F.Frozen(t["trial_hash"], t["parameters"])


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    p = build(tmp_path_factory, "fwd_base")
    run, block, start, end = F.forward_paths(p, frozen(p, 0), START)
    return {"p": p, "run": run, "block": block, "start": start, "end": end}


def test_the_forward_interval_starts_on_the_requested_date_and_ends_with_the_snapshot(base):
    p = base["p"]
    first = p.rm.eligible[base["start"] - 1]
    assert str(first.date()) >= START and str(p.rm.eligible[base["start"] - 2].date()) < START
    assert base["end"] == len(p.rm.eligible) and base["block"].n_decisions >= 2


def test_the_strategy_and_the_benchmarks_cover_the_same_intervals_from_the_opening_holdings(base):
    nav = base["run"].result.nav
    assert {len(b.nav) for b in base["run"].benchmarks.values()} == {len(nav)} and base["run"].benchmarks
    assert all(list(b.nav) == list(nav) for b in base["run"].benchmarks.values())


def test_only_the_forward_block_is_scored_not_the_valuations_before_its_start(base):
    out = F.forward_run(base["p"], frozen(base["p"], 0), START)
    nav = base["run"].result.nav
    owned = [s for s in base["block"].valuation_sessions if s in nav]
    assert out["intervals"] == len(owned) < len(nav) - 1, "the NAV holds pre-start valuations that must not be scored"
    assert min(owned) >= START
    assert set(out["diagnostics"]) >= {"strategy", F.PRIMARY_BENCHMARK} and all(
        len(d["largest_abs_interval_returns"]) == 3 for d in out["diagnostics"].values())
    for k, m in out["metrics"].items():
        assert m["intervals"] == len(owned) and -1.0 < m["total_return"] < 10.0, k


def test_the_summary_is_complete_and_passes_the_wording_rules(base):
    out = F.forward_run(base["p"], frozen(base["p"], 0), START)
    assert out["verdict"] in ("supports", "no_evidence", "against") and out["evidence_class"] == "synthetic_engineering_test"
    assert out["difference_vs_benchmark"]["ci_low"] <= out["difference_vs_benchmark"]["ci_high"]
    assert not F.forbidden_in(json.dumps(out))


def test_data_after_a_date_cannot_change_results_before_it(base, tmp_path_factory):
    q = build(tmp_path_factory, "fwd_perturbed", perturb_after=PERTURB_AFTER)
    run, *_ = F.forward_paths(q, frozen(q, 0), START)
    a, b = base["run"].result.nav, run.result.nav
    early = [s for s in a if s <= PERTURB_AFTER]
    assert len(early) > 20 and all(a[s] == b[s] for s in early)
    assert any(a[s] != b[s] for s in a if s > PERTURB_AFTER), "the perturbation must matter after its date (negative control)"


def test_the_frozen_selection_decides_the_path(base):
    other = frozen(base["p"], 3)
    assert other.winner_hash != frozen(base["p"], 0).winner_hash
    run, *_ = F.forward_paths(base["p"], other, START)
    assert any(run.result.nav[s] != base["run"].result.nav[s] for s in run.result.nav), "a different frozen selection must give a different path"


def test_a_snapshot_that_ends_before_the_start_is_refused(base):
    with pytest.raises(ValueError, match="no model row"):
        F.forward_start_position(base["p"], "2031-01-01")


def test_the_verdict_rule_is_applied_to_the_interval_only():
    assert F.verdict(0.1, 0.9) == "supports" and F.verdict(-0.5, 0.4) == "no_evidence" and F.verdict(-0.9, -0.1) == "against"
    assert F.verdict(float("nan"), 0.1) == "undefined"
    assert F.verdict(0.0, 1.0) == "no_evidence" and F.verdict(-1.0, 0.0) == "no_evidence"  # touching zero is not a result


def test_a_saved_copy_of_the_frozen_selection_gives_the_same_selection_as_the_finished_run_and_holds_nothing_else(base, tmp_path):
    t = base["p"].ev.design.trials[0]
    doc = {"frozen": True, "created_before_holdout_access": True, "trial_hash": t["trial_hash"], "parameters": t["parameters"],
           "something_else": {"not": "exported"}}
    run_dir = tmp_path / "finished"
    run_dir.mkdir()
    (run_dir / "final_selection.json").write_text(json.dumps(doc), encoding="utf-8")
    copy = tmp_path / "saved.json"
    assert F.export_selection(run_dir, copy) == frozen(base["p"], 0)
    assert F.load_selection(copy) == F.load_selection(run_dir)
    assert set(json.loads(copy.read_text(encoding="utf-8"))) == set(F.SELECTION_KEYS)
    unfrozen = tmp_path / "unfrozen.json"
    unfrozen.write_text(json.dumps({**doc, "frozen": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not marked frozen"):
        F.load_selection(unfrozen)
    (run_dir / "final_selection.json").write_text(json.dumps({**doc, "frozen": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not marked frozen"):
        F.export_selection(run_dir, tmp_path / "never.json")
    assert not (tmp_path / "never.json").exists()
