"""HMM layer: owned forward filter, alignment, EM export, the 8-step pipeline, the fixed regime table, and the
subprocess probe against the independent NumPy oracle (INV-HMM-003)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.regimes import (
    FixedRegimeTable,
    HMMParameters,
    align_states,
    filter_window,
    fit_em,
    forward_filter,
    log_emissions,
    run_hmm_pipeline,
    stationary_distribution,
    validate_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("a", "b")


def params(start=(0.6, 0.4), trans=((0.9, 0.1), (0.2, 0.8)), means=((0.0, 0.0), (1.0, -1.0)),
           variances=((1.0, 1.0), (0.5, 0.5))) -> HMMParameters:
    return HMMParameters(tuple(start), tuple(map(tuple, trans)), tuple(map(tuple, means)), tuple(map(tuple, variances)),
                         FEATURES, True, "test")


def sample(n=40, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)) * np.array([1.0, 0.7]) + np.array([0.2, -0.1])


def test_log_space_filter_agrees_with_the_likelihood_space_reference():
    p, y = params(), sample()
    got = forward_filter(p, y).filtered
    want = filter_window(np.array(p.start), np.array(p.transition), np.exp(log_emissions(p, y)))
    np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-12)


def test_log_space_filter_survives_underflow_that_breaks_likelihood_space():
    p, y = params(), sample(10) * 1000  # likelihoods underflow to exactly zero
    assert np.all(np.exp(log_emissions(p, y)) == 0.0)
    f = forward_filter(p, y).filtered
    assert np.all(np.isfinite(f)) and np.allclose(f.sum(axis=1), 1.0)


def test_continuing_from_the_boundary_equals_filtering_the_concatenated_sequence():
    p, y = params(), sample(30)
    whole = forward_filter(p, y).filtered
    head = forward_filter(p, y[:20])
    tail = forward_filter(p, y[20:], prior=head.filtered[-1], first_is_start=False)
    np.testing.assert_allclose(tail.filtered, whole[20:], atol=1e-12)


def test_filtered_posteriors_depend_only_on_past_and_present_rows():
    p, y = params(), sample(30)
    base = forward_filter(p, y).filtered
    y2 = y.copy()
    y2[15:] += 3.0
    np.testing.assert_array_equal(base[:15], forward_filter(p, y2).filtered[:15])  # no backward pass, no smoothing
    assert not np.allclose(base[15:], forward_filter(p, y2).filtered[15:])


def test_first_oos_prediction_comes_from_the_boundary_not_from_stationarity():
    p, y = params(), sample(40)
    r = run_hmm_pipeline(y[:25], y[25:], n_states=2, seed=1, max_iter=10, tol=1e-6, feature_order=FEATURES, raw_override=p)
    a = np.array(r.aligned.transition)
    np.testing.assert_allclose(r.first_oos_predicted, r.boundary_posterior @ a, atol=1e-14)
    assert not np.allclose(r.first_oos_predicted, stationary_distribution(a) @ a, atol=1e-4)
    np.testing.assert_array_equal(r.boundary_posterior, r.train.filtered[-1])


def test_alignment_orders_by_volatility_then_mean_then_index_and_permutes_every_parameter():
    p = params(start=(0.7, 0.3), means=((5.0, 5.0), (0.0, 0.0)), variances=((2.0, 2.0), (0.1, 0.1)))
    a, perm = align_states(p)
    assert perm == (1, 0)
    assert a.start == (0.3, 0.7) and a.means[0] == (0.0, 0.0) and a.variances[0] == (0.1, 0.1)
    assert a.transition[0][1] == p.transition[1][0] and a.transition[1][1] == p.transition[0][0]
    tie = params(means=((1.0, 1.0), (1.0, 1.0)), variances=((1.0, 1.0), (1.0, 1.0)))
    assert align_states(tie)[1] == (0, 1)  # full tie -> original index
    mean_tie = params(means=((2.0, 2.0), (1.0, 1.0)), variances=((1.0, 1.0), (1.0, 1.0)))
    assert align_states(mean_tie)[1] == (1, 0)  # equal volatility -> lower mean first


@pytest.mark.parametrize("bad", [
    params(trans=((0.9, 0.2), (0.2, 0.8))), params(variances=((1.0, 1.0), (1e-9, 1.0))),
    params(means=((0.0, np.nan), (1.0, 1.0))), params(trans=((1.0, 0.0), (-0.1, 1.1)))])
def test_validation_rejects_bad_parameters(bad):
    with pytest.raises(ValueError):
        validate_parameters(bad)


def test_feature_count_and_order_are_checked():
    with pytest.raises(ValueError):
        validate_parameters(params(), n_features=3)
    with pytest.raises(ValueError):
        log_emissions(params(), np.zeros((4, 3)))


def test_parameter_json_round_trip_is_exact():
    p = params()
    assert HMMParameters.from_json(p.to_json()) == p


def test_em_is_deterministic_and_recovers_a_volatility_split():
    rng = np.random.default_rng(11)
    y = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(0, 2.0, (60, 2)), rng.normal(0, 0.3, (60, 2))])
    a = fit_em(y, 2, seed=5, max_iter=200, tol=1e-6, feature_order=FEATURES)
    b = fit_em(y, 2, seed=5, max_iter=200, tol=1e-6, feature_order=FEATURES)
    assert a == b and a.converged
    aligned, _ = align_states(a)
    vols = np.sqrt(np.array(aligned.variances).mean(axis=1))
    assert vols[0] < 0.6
    assert vols[1] > 1.5
    r = run_hmm_pipeline(y, None, n_states=2, seed=5, max_iter=200, tol=1e-6, feature_order=FEATURES)
    assert r.train.filtered[:50, 0].mean() > 0.8 and r.train.filtered[70:110, 1].mean() > 0.8
    assert r.oos is None and r.first_oos_predicted is None


def test_invalid_fitted_start_uses_a_reported_stationary_fallback():
    p = params(start=(float("nan"), 0.4))
    r = run_hmm_pipeline(sample(20), None, n_states=2, seed=1, max_iter=5, tol=1e-6, feature_order=FEATURES, raw_override=p)
    assert r.start_fallback_stationary
    np.testing.assert_allclose(r.aligned.start, stationary_distribution(np.array(r.aligned.transition)), atol=1e-12)
    good = run_hmm_pipeline(sample(20), None, n_states=2, seed=1, max_iter=5, tol=1e-6, feature_order=FEATURES, raw_override=params())
    assert not good.start_fallback_stationary


def test_fixed_regime_table_loads_and_looks_up_as_of_decision_cutoff():
    t = FixedRegimeTable.load(ROOT / "data/fixtures/fixture_3asset/fixed_regime_table.csv")
    probs, label = t.asof(pd.Timestamp("2024-01-31T21:15:00Z"))
    assert label == "low_risk" and probs.sum() == pytest.approx(1.0) and probs[0] > probs[1]
    _, label = t.asof(pd.Timestamp("2024-02-12T12:00:00Z"))  # between rows: the latest available row
    assert label in {"low_risk", "high_risk"}
    with pytest.raises(LookupError):
        t.asof(pd.Timestamp("2024-01-02T21:14:59Z"))  # one second before the first row is available
    assert t.asof(pd.Timestamp("2024-01-02T21:15:00Z"))[1] == "low_risk"


def test_fixed_regime_table_rejects_bad_schema_and_probabilities(tmp_path):
    good = (ROOT / "data/fixtures/fixture_3asset/fixed_regime_table.csv").read_text(encoding="utf-8")
    cases = {"cols": good.replace("state_0_probability", "p0"), "sum": good.replace("0.9,0.1", "0.9,0.3", 1),
             "label": good.replace("low_risk", "medium_risk", 1)}
    for name, text in cases.items():
        f = tmp_path / f"{name}.csv"
        f.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            FixedRegimeTable.load(f)


def test_probe_in_a_fresh_subprocess_agrees_with_the_independent_oracle(tmp_path):
    out = tmp_path / "probe.json"
    subprocess.run([sys.executable, "-m", "quant_pipeline.hmm_probe", "--out", str(out), "--root", str(ROOT)], check=True,
                   capture_output=True, cwd=ROOT)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "hmm_probe_oracle.py"), str(out)], capture_output=True, text=True,
                       check=False)
    report = json.loads(r.stdout)
    assert r.returncode == 0 and report["status"] == "pass", r.stdout
    assert report["nonidentity_permutation_exercised"] and report["rows_from_mutation_changed"]
    assert report["stationary_reset_would_differ"] and report["rows_before_mutation_unchanged"] == 0.0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["em"]["converged"] and d["em"]["n_train_rows"] == 60 and not d["start_fallback_stationary"]
    assert max(v for k, v in report.items() if isinstance(v, float)) <= 1e-10


def test_oracle_actually_detects_a_corrupted_probe(tmp_path):
    out = tmp_path / "probe.json"
    subprocess.run([sys.executable, "-m", "quant_pipeline.hmm_probe", "--out", str(out), "--root", str(ROOT)], check=True,
                   capture_output=True, cwd=ROOT)
    d = json.loads(out.read_text(encoding="utf-8"))
    d["boundary_posterior"][0] += 1e-6  # a tiny production-side discrepancy must be caught
    out.write_text(json.dumps(d), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "hmm_probe_oracle.py"), str(out)], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 1 and json.loads(r.stdout)["status"] == "fail"
