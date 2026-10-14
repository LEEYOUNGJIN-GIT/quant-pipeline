"""Bootstrap indices, Sobol trial enumeration, registry and selection (spec 11.3, 11.4, 11.7).
Production output is compared with the independent oracle files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline import trials as T
from quant_pipeline.bootstrap import index_hash, index_matrix, run_bootstrap
from quant_pipeline.config import load_config

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests" / "expected" / "fixture_3asset" / "oracle"
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")


# ----------------------------------------------------------------------------------------------- bootstrap
def fixture_matrix():
    b = CFG.bootstrap
    return index_matrix(b.expected_oos_rows, b.replicates, b.mean_block_length, CFG.reproducibility.random_streams.bootstrap)


def test_bootstrap_index_hash_equals_the_oracle_exactly():
    m = fixture_matrix()
    assert m.shape == (200, 22) and m.min() >= 0 and m.max() <= 21
    assert index_hash(m) + "\n" == (ORACLE / "bootstrap_indices.sha256").read_text(encoding="utf-8")


def test_bootstrap_is_deterministic_and_seed_sensitive():
    assert np.array_equal(fixture_matrix(), fixture_matrix())
    assert index_hash(index_matrix(22, 200, 5, 1)) != index_hash(index_matrix(22, 200, 5, 2))


def test_stationary_blocks_have_the_configured_mean_length():
    m = index_matrix(500, 200, 5, 7)
    continues = ((m[:, 1:] - m[:, :-1]) % 500 == 1).mean()
    assert continues == pytest.approx(1 - 1 / 5, abs=0.02)  # restart probability 1 / mean_block_length (a restart can also be +1)


@pytest.mark.parametrize("args", [(1, 10, 5, 0), (22, 0, 5, 0), (22, 10, 0, 0)])
def test_invalid_bootstrap_dimensions_are_rejected(args):
    with pytest.raises(ValueError):
        index_matrix(*args)


def run(series, rf, **over):
    b = CFG.bootstrap
    kw = {"replicates": b.replicates, "mean_block_length": b.mean_block_length, "seed": CFG.reproducibility.random_streams.bootstrap,
          "confidence": b.confidence_level, "periods_per_year": 252, "minimum_rows": b.minimum_oos_rows, "expected_rows": 22,
          "inferential": b.intervals_are_inferential} | over
    return run_bootstrap(series, rf, **kw)


def test_the_aligned_tuple_is_resampled_with_one_shared_index_matrix():
    rng = np.random.default_rng(2)
    rf = np.full(22, 0.0001)
    strat, bench = rng.normal(0.001, 0.01, 22), rng.normal(0.0005, 0.01, 22)
    r = run({"strategy": strat, "benchmark": bench}, rf)
    assert r.status == "ok" and r.index_hash == index_hash(fixture_matrix()) and r.indices.shape == (200, 22)
    assert r.statistics["strategy"]["replicates"] == 200 and not r.inferential  # fixture bootstrap is non-inferential smoke
    for s in r.statistics.values():
        assert s["ci_low"] <= s["ci_high"]


def test_a_series_with_zero_excess_variance_is_reported_undefined_not_crashed_or_zeroed():
    rf = np.full(22, 0.0002)
    r = run({"cash": rf.copy(), "real": np.random.default_rng(4).normal(0.0005, 0.01, 22)}, rf)
    assert r.statistics["cash"]["undefined"] == 1.0 and np.isnan(r.statistics["cash"]["ci_low"]) and np.isnan(r.statistics["cash"]["point_sharpe"])
    assert r.statistics["real"]["undefined"] == 0.0 and np.isfinite(r.statistics["real"]["ci_high"])


def test_bootstrap_refuses_wrong_length_misaligned_or_short_input():
    rf = np.zeros(22)
    with pytest.raises(ValueError, match="expected 22"):
        run({"s": np.zeros(20)}, np.zeros(20))
    with pytest.raises(ValueError, match="aligned"):
        run({"s": np.zeros(21)}, rf)
    short = run({"s": np.zeros(5)}, np.zeros(5), expected_rows=None, minimum_rows=22)
    assert short.status == "insufficient_sample" and short.indices.size == 0  # reported, never silently skipped


# ----------------------------------------------------------------------------------------------- Sobol trials
def test_trial_enumeration_equals_the_oracle_points():
    d = T.enumerate_trials(CFG)
    o = pd.read_csv(ORACLE / "sobol_points.csv", float_precision="round_trip")
    meta = json.loads((ORACLE / "sobol_meta.json").read_text(encoding="utf-8"))
    assert list(d.dimension_order) == meta["dimension_order"] and d.cardinality == meta["cardinality"] == 8
    assert [tuple(p["mapped"]) for p in d.points] == [tuple(r) for r in o[[c for c in o.columns if c.startswith("idx_")]].values.tolist()]
    assert [p["trial_hash"] for p in d.points] == o.trial_hash.tolist()
    ucols = [c for c in o.columns if c.startswith("u_")]
    assert all(p["u"][j] == o.iloc[i][c] for i, p in enumerate(d.points) for j, c in enumerate(ucols))
    assert d.stop_reason == meta["stop_reason"] == "cap_reached" and d.seed == 1729


def test_eight_distinct_trials_cover_every_combination_of_the_three_binary_dimensions():
    d = T.enumerate_trials(CFG)
    assert len(d.trials) == 8 and len({t["trial_hash"] for t in d.trials}) == 8
    combos = {tuple(t["parameters"][k] for k in d.dimension_order) for t in d.trials}
    assert len(combos) == 8
    for t in d.trials:  # singleton dimensions are part of the resolved parameters
        assert set(t["parameters"]) == set(T.TRIAL_DIMENSIONS)
    assert [t["point_index"] for t in d.trials] == sorted(t["point_index"] for t in d.trials)  # Sobol-index order


def test_the_cap_limits_distinct_trials_and_the_design_hash_is_stable():
    cfg = CFG.model_copy(update={"evaluation": CFG.evaluation.model_copy(update={"max_parameter_trials": 3})})
    d3 = T.enumerate_trials(cfg)
    assert d3.target == 3 and len(d3.trials) == 3 and d3.stop_reason == "cap_reached"
    assert T.enumerate_trials(CFG).design_hash == T.enumerate_trials(CFG).design_hash != d3.design_hash


def test_a_finite_space_smaller_than_the_cap_is_exhausted_not_searched_forever():
    agents = CFG.agents.model_copy(update={"defensive": CFG.agents.defensive.model_copy(update={"overlay_gamma_candidates": [0.75, 1.0]})})
    obj = CFG.objective.model_copy(update={"lambda_track_candidates": [1.0], "lambda_cost_candidates": [1.0]})
    d = T.enumerate_trials(CFG.model_copy(update={"agents": agents, "objective": obj}))
    assert d.cardinality == 2 and len(d.trials) == 2 and d.stop_reason == "candidate_space_exhausted"


def test_duplicate_mapped_configurations_do_not_consume_the_cap(monkeypatch):
    lo, hi = 0.1, 0.9
    seq = [(lo, lo, lo), (lo, lo, lo), (lo, lo, hi), (lo, lo, hi), (lo, hi, lo), (lo, hi, hi), (hi, lo, lo), (lo, lo, lo),
           (hi, lo, hi), (hi, hi, lo), (hi, hi, hi)]  # 11 points, 8 distinct, 3 duplicates

    class Scripted:
        def __init__(self, *a, **k):
            self.i = 0

        def random(self, n):
            self.i += 1
            return np.array([seq[self.i - 1]])

    monkeypatch.setattr(T.qmc, "Sobol", Scripted)
    d = T.enumerate_trials(CFG)
    dup = [p for p in d.points if not p["counts_toward_cap"]]
    assert len(d.points) == 11 and len(dup) == 3 and all(p["duplicate_of_point"] is not None for p in dup)
    assert len(d.trials) == 8 and len({t["trial_hash"] for t in d.trials}) == 8  # duplicates are recorded but not counted
    assert d.points[-1]["counts_toward_cap"] and d.stop_reason == "cap_reached"


def test_a_sampler_that_cannot_reach_the_target_fails_loudly(monkeypatch):
    class Stuck:
        def __init__(self, *a, **k): ...
        def random(self, n):
            return np.full((n, 3), 0.1)

    monkeypatch.setattr(T.qmc, "Sobol", Stuck)
    with pytest.raises(T.SamplerFailure):
        T.enumerate_trials(CFG)


def test_registry_is_append_only_and_enforces_the_cap():
    reg = T.TrialRegistry("run", cap=2)
    reg.register("f", "h1", "completed", score=1.0)
    with pytest.raises(ValueError, match="append-only"):
        reg.register("f", "h1", "completed")
    reg.register("f", "h2", "pruned")  # pruned / failed / resumed all count toward the cap
    with pytest.raises(RuntimeError, match="cap"):
        reg.register("f", "h3", "completed")
    reg.register("g", "h1", "completed")  # the cap is per fold


# ----------------------------------------------------------------------------------------------- selection
def test_selection_uses_mean_score_and_reports_margin():
    s = T.select({"a": [1.0, 3.0], "b": [2.5, 2.5], "c": [0.0, 1.0]}, dict.fromkeys("abc", 0.1), dict.fromkeys("abc", 1.0), 1e-12)
    assert s.winner == "b" and s.ranking == ("b", "a", "c") and s.tie_path == ("score",)
    assert s.margin == pytest.approx(0.5) and s.mean_scores["b"] == 2.5


def test_ties_break_on_turnover_then_complexity_then_hash():
    sc = {"x": [1.0], "y": [1.0 + 1e-14], "z": [1.0]}
    by_turn = T.select(sc, {"x": 0.3, "y": 0.2, "z": 0.4}, dict.fromkeys("xyz", 1.0), 1e-12)
    assert by_turn.winner == "y" and by_turn.tie_path == ("score", "turnover")
    by_cx = T.select(sc, dict.fromkeys("xyz", 0.2), {"x": 3.0, "y": 2.0, "z": 1.0}, 1e-12)
    assert by_cx.winner == "z" and by_cx.tie_path[-1] == "complexity"
    by_hash = T.select(sc, dict.fromkeys("xyz", 0.2), dict.fromkeys("xyz", 1.0), 1e-12)
    assert by_hash.winner == "x" and by_hash.tie_path[-1] == "trial_hash"  # smallest hash wins


def test_scores_outside_the_tolerance_are_not_tied():
    s = T.select({"a": [1.0], "b": [1.0 + 1e-6]}, {"a": 0.0, "b": 9.0}, dict.fromkeys("ab", 0.0), 1e-12)
    assert s.winner == "b"  # a 1e-6 edge is real; turnover is not consulted


def test_ineligible_and_non_finite_trials_are_excluded():
    s = T.select({"a": [9.0], "b": [1.0], "c": [float("nan")]}, dict.fromkeys("abc", 0.0), dict.fromkeys("abc", 0.0), 1e-12,
                 eligibility={"a": False, "b": True, "c": True})
    assert s.winner == "b" and "c" not in s.mean_scores
    with pytest.raises(ValueError):
        T.select({"a": [float("nan")]}, {"a": 0.0}, {"a": 0.0}, 1e-12)


def test_selection_is_independent_of_dict_order_and_parallel_completion_order():
    sc = {"a": [1.0, 2.0], "b": [1.5, 1.5], "c": [0.5, 0.5]}
    t, c = dict.fromkeys("abc", 0.1), dict.fromkeys("abc", 1.0)
    first = T.select(sc, t, c, 1e-12)
    for order in (["c", "b", "a"], ["b", "a", "c"]):
        assert T.select({k: sc[k] for k in order}, t, c, 1e-12).ranking == first.ranking
