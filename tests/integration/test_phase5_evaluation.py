"""Nested evaluation gate (spec 11, 14, 16.11): 48 trial-fold evaluations, deterministic selection, continuous outer OOS,
cost scenarios and benchmarks, with the holdout untouched. Evidence class: fixture_path."""
from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_phase5_engine as E

from quant_pipeline.agents import diversification as DIV
from quant_pipeline.bootstrap import index_hash, run_bootstrap
from quant_pipeline.decision import TrialParams
from quant_pipeline.evaluation import BENCHMARK_IDS, Evaluator, interval_returns, net_excess_sharpe
from quant_pipeline.splits import build_split_manifest
from quant_pipeline.trials import to_trial_params

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = build_split_manifest(E.CFG, E.CALS, E.RM)
MEASURE: dict[str, dict] = {}  # auxiliary-solve counters bracketing exactly this module's selection + outer runs


@pytest.fixture(scope="module")
def ev():
    return Evaluator(E.CFG, E.MD, E.CALS, E.SESS, E.RM, E.REGIME, E.HASHES, MANIFEST, E.OPEN)


@pytest.fixture(scope="module")
def selection(ev):
    MEASURE["before"] = dict(DIV.STATS)  # other test modules may already have solved in this process; bracket our own work
    outcomes, paths = ev.run_selection()
    return outcomes, paths


@pytest.fixture(scope="module")
def outer(ev, selection):
    runs = {m: ev.run_outer(selection[0], m) for m in (0.0, 1.0, 2.0)}
    MEASURE["after"] = dict(DIV.STATS)
    return runs


def test_all_forty_eight_trial_fold_evaluations_execute_and_are_registered(ev, selection):
    outcomes, _ = selection
    assert [o.name for o in outcomes] == ["outer-1", "outer-2", "final"] and ev.counts.trial_fold_evaluations == 48
    assert all(len(o.evaluations) == 16 for o in outcomes)
    assert len(ev.registry.rows) == 48 and len({k[2] for k in ev.registry.rows}) == 8  # 3 passes x 2 folds x 8 candidates
    for o in outcomes:
        assert {e.trial_hash for e in o.evaluations} == {t["trial_hash"] for t in ev.design.trials}
        assert {e.opening_state_hash for e in o.evaluations} == {e.opening_state_hash for e in o.evaluations[:1]}  # one common anchor


def test_prefix_sharing_executes_far_fewer_decisions_than_the_no_sharing_bound(ev, selection):
    c = ev.counts
    assert (c.executed_paths, c.executed_decisions) == (8, 480) and c.no_sharing_replayed_decisions == 8 * (34 + 40 + 44 + 50 + 54 + 60)
    assert c.executed_decisions < c.no_sharing_replayed_decisions  # reported both ways (spec 3.4: the cap uses the upper bound)


def test_the_shared_prefix_equals_a_truly_isolated_path_to_each_endpoint(ev, selection):
    """The sharing optimization is only valid if an isolated path to an endpoint is identical to the prefix of the long path."""
    _, paths = selection
    t = ev.design.trials[3]
    h = t["trial_hash"]
    label = f"inner:{h[7:19]}"
    isolated = ev.engine.run(label=label, fold_id=label, trial=to_trial_params(E.CFG, t["parameters"]), trial_hash=h, multiplier=1.0,
                             start=1, end=34, opening=E.OPEN)
    long = paths[h]
    assert [d.plan_hash for d in isolated.decisions] == [d.plan_hash for d in long.decisions[:34]]
    anchors = [s for s in long.nav if s <= isolated.decisions[-1].cutoff_utc[:10]]
    assert {s: isolated.nav[s] for s in anchors} == {s: long.nav[s] for s in anchors}


def test_every_candidate_is_scored_from_its_own_ledger_with_a_hand_checkable_metric(ev, selection):
    outcomes, paths = selection
    e = outcomes[0].evaluations[5]
    blk = MANIFEST.passes[0].inner[e.inner_index - 1].validation
    rets, rf = interval_returns(paths[e.trial_hash].nav), ev.risk_free()
    ex = np.array([rets[s] - rf[s] for s in blk.valuation_sessions])
    assert e.score == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252), rel=1e-12)
    assert net_excess_sharpe(np.array([rets[s] for s in blk.valuation_sessions]), np.array([rf[s] for s in blk.valuation_sessions]), 252) == e.score
    assert e.scored_intervals == blk.n_valuation_intervals and e.censored_decision == blk.end_pos


def test_selection_is_deterministic_across_processes(selection):
    outcomes, _ = selection
    mine = {o.name: [o.winner_hash, [round(e.score, 12) for e in o.evaluations]] for o in outcomes}
    code = ("import sys, json; sys.path.insert(0, 'tests/integration'); import test_phase5_evaluation as t;"
            "ev = t.Evaluator(t.E.CFG, t.E.MD, t.E.CALS, t.E.SESS, t.E.RM, t.E.REGIME, t.E.HASHES, t.MANIFEST, t.E.OPEN);"
            "o, _ = ev.run_selection();"
            "print(json.dumps({x.name: [x.winner_hash, [round(e.score, 12) for e in x.evaluations]] for x in o}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == json.loads(json.dumps(mine))  # same registry scores, same winners


def test_each_pass_selects_a_member_of_the_design_with_persisted_ranking_and_margin(ev, selection):
    outcomes, _ = selection
    members = {t["trial_hash"] for t in ev.design.trials}
    for o in outcomes:
        assert o.winner_hash in members and o.selection.ranking[0] == o.winner_hash and len(o.selection.ranking) <= 8
        assert o.selection.tie_path[0] == "score" and np.isfinite(o.selection.margin) and o.selection.margin >= 0
        assert len(o.leave_one_fold_out_winners) == 2  # stability diagnostic; a disagreement is labelled, not hidden
        assert o.selection_unstable == (len(set(o.leave_one_fold_out_winners)) > 1)


def test_known_gap_every_evaluation_meets_the_fixture_minima(selection):
    """Spec 14 requires >= 4 decisions, >= 4 realized fills and >= 4 intervals in all 48 evaluations (OQ-031)."""
    outcomes, _ = selection
    bad = [(o.name, e.trial_hash[7:13], e.inner_index, e.ineligible_reason) for o in outcomes for e in o.evaluations if not e.eligible]
    assert bad == [("outer-2", "3cbe90", 2, "fills")]  # the measured, exact set of failures
    assert len(bad) == 1 and sum(e.eligible for o in outcomes for e in o.evaluations) == 47


@pytest.mark.xfail(strict=True, reason="OQ-031: one of 48 evaluations has 3 realized fills (<4); thresholds are not adapted")
def test_acceptance_all_forty_eight_evaluations_are_eligible(selection):
    assert all(e.eligible for o in selection[0] for e in o.evaluations)


def test_outer_oos_is_one_continuous_ledger_with_carry_and_origin_ids(ev, outer):
    r1, r2 = outer[1.0]
    assert [r.fold_id for r in (r1, r2)] == ["outer-1", "outer-2"]
    assert max(r1.result.nav) == "2024-04-17" and min(r2.result.nav) == "2024-04-18"  # contiguous, no reset, no overlap
    crossing = [f for f in r2.result.fills if f.decision_position == 50]
    assert crossing and all(f.origin["fold_id"] == "outer-1" and f.origin["trial_hash"] == r1.trial_hash for f in crossing)
    assert {d.cutoff_utc[:10] for d in r2.result.decisions} and r2.result.decisions[0].position == 51


def test_each_outer_fold_uses_only_its_own_frozen_winner(ev, selection, outer):
    outcomes, _ = selection
    win = {o.name: o.winner_hash for o in outcomes}
    r1, r2 = outer[1.0]
    assert (r1.trial_hash, r2.trial_hash) == (win["outer-1"], win["outer-2"])
    assert win["final"] not in (None, "")  # chosen on pre-holdout history only and frozen for the (unopened) holdout


def test_the_oos_return_tuple_is_the_22_interval_aligned_set_without_the_holdout(ev, outer):
    ret = ev.oos_returns(outer[1.0])
    assert {k: len(v) for k, v in ret.items()} == {"strategy": 22, "rf": 22, **dict.fromkeys(BENCHMARK_IDS, 22)}
    assert all(np.isfinite(v).all() for v in ret.values())
    used = {s for r in outer[1.0] for s in r.result.nav}
    assert not used & set(MANIFEST.holdout.valuation_sessions)  # the sealed holdout was never touched


def test_cost_scenarios_share_forecasts_but_have_their_own_ledgers(outer):
    a = {m: [d.a_bar for r in outer[m] for d in r.result.decisions] for m in outer}
    assert a[0.0] == a[1.0] == a[2.0]  # common targets / normalization scales
    costs = {m: sum(r.result.costs for r in outer[m]) for m in outer}
    assert costs[0.0] == 0 and costs[1.0] > 0 and costs[2.0] > 0
    navs = {m: tuple(v for r in outer[m] for v in r.result.nav.values()) for m in outer}
    assert len(set(navs.values())) == 3
    assert any(d0.weights != d1.weights for r0, r1 in zip(outer[0.0], outer[1.0], strict=True)
               for d0, d1 in zip(r0.result.decisions, r1.result.decisions, strict=True))  # nonempty cost-sensitive intentions


def test_benchmarks_are_labelled_and_run_on_the_same_timeline(ev, outer):
    r = outer[1.0][0]
    assert set(r.benchmarks) == set(BENCHMARK_IDS)
    assert all(list(b.nav) == list(r.result.nav) for b in r.benchmarks.values())
    ref = outer[1.0][0].benchmarks["equal_weight_buy_and_hold_reference"]
    assert ref.decisions[0].one_way_turnover_intended > E.CFG.constraints.turnover_one_way_limit  # only the reference is exempt
    assert max(d.one_way_turnover_intended for d in outer[1.0][0].benchmarks["equal_weight_periodic"].decisions) <= 0.5 + 1e-9


def test_bootstrap_consumes_exactly_the_22_aligned_intervals_with_the_oracle_index_hash(ev, outer):
    ret = ev.oos_returns(outer[1.0])
    b = E.CFG.bootstrap
    res = run_bootstrap({k: v for k, v in ret.items() if k != "rf"}, ret["rf"], replicates=b.replicates,
                        mean_block_length=b.mean_block_length, seed=E.CFG.reproducibility.random_streams.bootstrap,
                        confidence=b.confidence_level, periods_per_year=252, minimum_rows=b.minimum_oos_rows,
                        expected_rows=b.expected_oos_rows, inferential=b.intervals_are_inferential)
    oracle = (ROOT / "tests/expected/fixture_3asset/oracle/bootstrap_indices.sha256").read_text(encoding="utf-8").strip()
    assert res.status == "ok" and res.indices.shape == (200, 22) and res.index_hash == oracle == index_hash(res.indices)
    assert not res.inferential and set(res.statistics) == {"strategy", *BENCHMARK_IDS}
    cash = res.statistics["cash_krw"]  # cash minus its own risk-free leg is identically zero: undefined, reported, not coerced
    assert cash["undefined"] == 1.0 and np.isnan(cash["ci_low"]) and cash["replicates"] == 0.0
    assert all(res.statistics[k]["undefined"] == 0.0 and res.statistics[k]["replicates"] == 200 for k in BENCHMARK_IDS[1:])


def test_the_strategy_keeps_cash_nonnegative_and_reconciles_on_every_outer_fill(outer):
    for r in outer[1.0]:
        assert all(x["within_tolerance"] for x in r.result.reconciliation)
        assert (r.result.ledger_rows.cash_krw.map(Decimal) >= 0).all()
    assert TrialParams.first(E.CFG) is not None


def test_executed_work_never_exceeds_the_graph_derived_plan(ev, selection, outer):
    """A plan is a valid ceiling only if the real run stays inside it (selection and outer folds; the holdout is not run here)."""
    from quant_pipeline.plan import build_compute_plan
    from quant_pipeline.trials import enumerate_trials

    plan = build_compute_plan(E.CFG, MANIFEST, enumerate_trials(E.CFG), E.HASHES)["counts"]
    _, paths = selection
    allocation = sum(len(p.decisions) for p in paths.values()) + sum(len(r.result.decisions) for runs in outer.values() for r in runs)
    aux = MEASURE["after"]["problems"] - MEASURE["before"]["problems"]
    caches = ev.engine.caches.counts
    assert allocation <= plan["allocation_optimizer_solve"]["deduplicated"]
    assert caches["fits"] <= plan["model_fit_unit"]["deduplicated"]
    assert caches["calibrations"] <= plan["calibration_unit"]["deduplicated"]
    assert 0 < aux <= plan["auxiliary_optimizer_solve"]["deduplicated"]
    assert ev.counts.executed_decisions <= plan["allocation_optimizer_solve"]["no_sharing_upper_bound"]


def test_both_outer_tests_meet_their_decision_fill_and_interval_minima(outer):
    import pandas as pd

    need = E.CFG.evaluation.outer
    for run, fold in zip(outer[1.0], MANIFEST.outer, strict=True):
        b = fold.test
        lo, hi = pd.Timestamp(b.first_decision_cutoff_utc), pd.Timestamp(b.fence_utc)
        fills = [f for f in run.result.fills if f.status in ("filled", "reduced_cash", "reduced_holdings") and lo < pd.Timestamp(f.ts) <= hi]
        assert len(run.result.decisions) >= need.minimum_decisions_per_test, fold.fold_id
        assert len(fills) >= need.minimum_fills_per_test, fold.fold_id
        assert b.n_valuation_intervals >= need.minimum_scored_nav_intervals_per_test, fold.fold_id
