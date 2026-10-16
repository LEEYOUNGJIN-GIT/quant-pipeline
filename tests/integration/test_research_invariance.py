"""Future-data invariance through the whole research path (spec 12.6 look-ahead, INV-LEAK/INV-OOS style; INV-DATA-001), synthetic data.

Two snapshots are identical up to a session date T; after T every price return, FX quote and rate is deterministically rewritten in the second.
The same trial path (fits, PCA, covariance, HMM, calibration, agents, optimizer, plan, ledger, valuation) is run on both. Everything decided,
executed or valued at or before the decision cutoff c* must be bit-identical; the path after c* must differ, so the test is not vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline.research_pipeline import ResearchPipeline
from support.research_smoke import SMOKE_WINDOW, write_smoke_config
from support.synthetic_research import build_research_repo

DECISION_INDEX = 9  # the tenth scheduled decision: about half of the 21 decisions lie after it


def world(tmp_path: Path, perturb_after: str | None):
    root, _ = build_research_repo(tmp_path, start=SMOKE_WINDOW[0], end=SMOKE_WINDOW[1], perturb_after=perturb_after)
    cfg = write_smoke_config(root)
    p = ResearchPipeline(root, root / "out", "inv", config=cfg, allow_synthetic=True)
    trial = p.ev.design.trials[0]
    return p, trial, p.ev.run_trial_path(trial)


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    base_p, trial, base = world(tmp_path_factory.mktemp("base"), None)
    dec = base_p.rm.decision_rows[DECISION_INDEX]
    assert dec.on_complete_case_row  # a clean boundary: the decision's own row is observed at its cutoff
    t_date = dec.session.strftime("%Y-%m-%d")
    pert_p, _, pert = world(tmp_path_factory.mktemp("pert"), t_date)
    cutoff = base_p.ev.engine.decision_cutoffs[dec.position]
    return {"base_p": base_p, "pert_p": pert_p, "trial": trial, "base": base, "pert": pert, "t_date": t_date, "cutoff": cutoff, "dec": dec}


def test_the_two_snapshots_differ_only_after_the_boundary(worlds):
    b, q = worlds["base_p"].md, worlds["pert_p"].md
    cut = np.datetime64(worlds["cutoff"].tz_convert("UTC").tz_localize(None))
    for a in b.total_return:
        sb, sq = b.total_return[a], q.total_return[a]
        early = sb.available <= cut
        assert np.array_equal(sb.value[early], sq.value[: len(sb.value)][early])  # identical through the boundary
        assert not np.array_equal(sb.value[~early], sq.value[~early])  # rewritten after it
    assert worlds["base_p"].snapshot.snapshot_id != worlds["pert_p"].snapshot.snapshot_id


def test_every_decision_up_to_the_boundary_is_identical(worlds):
    cut = worlds["cutoff"]
    before = [(a, b) for a, b in zip(worlds["base"].decisions, worlds["pert"].decisions, strict=True) if pd.Timestamp(a.cutoff_utc) <= cut]
    assert len(before) == DECISION_INDEX + 1
    for a, b in before:
        assert a.position == b.position and a.cutoff_utc == b.cutoff_utc
        assert a.plan_hash == b.plan_hash, a.position
        assert a.a_bar == b.a_bar and a.weights == b.weights and a.w_minus == b.w_minus
        assert a.branch == b.branch and a.n_orders == b.n_orders and a.objective_terms == b.objective_terms


def test_the_hmm_probabilities_and_fits_used_up_to_the_boundary_are_identical(worlds):
    ep, eq = worlds["base_p"].ev.engine, worlds["pert_p"].ev.engine
    cut = worlds["cutoff"]
    seen = 0
    for pos, c in ep.decision_cutoffs.items():
        if c > cut:
            continue
        rpos = ep.refit_position(pos)
        params = _params(worlds)
        fb, fq = ep.fit(rpos, params.n_components), eq.fit(rpos, params.n_components)
        # the snapshots have different manifests by construction, so compare the fingerprint without that field: it hashes the observation set the fit saw
        strip = lambda fp: {k: v for k, v in fp.items() if k != "data_manifest_hash"}
        assert strip(fb.fingerprint) == strip(fq.fingerprint)
        assert np.array_equal(fb.covariance.matrix, fq.covariance.matrix) and np.array_equal(fb.pca.loadings, fq.pca.loadings)
        tb, _ = ep.target(pos, params, fb)
        tq, _ = eq.target(pos, params, fq)
        assert tb.regime_probabilities == tq.regime_probabilities, pos
        seen += 1
    assert seen >= 5


def _params(w):
    from quant_pipeline.trials import to_trial_params

    return to_trial_params(w["base_p"].cfg, w["trial"]["parameters"])


def test_fills_and_valuations_up_to_the_boundary_are_identical(worlds):
    cut = worlds["cutoff"]
    fb = [(f.order_id, f.ts, f.realized_quantity, f.status) for f in worlds["base"].fills if pd.Timestamp(f.ts) <= cut]
    fp = [(f.order_id, f.ts, f.realized_quantity, f.status) for f in worlds["pert"].fills if pd.Timestamp(f.ts) <= cut]
    assert fb == fp and len(fb) > 0
    nb = {k: v for k, v in worlds["base"].nav.items() if k <= worlds["t_date"]}
    npert = {k: v for k, v in worlds["pert"].nav.items() if k <= worlds["t_date"]}
    assert nb == npert and len(nb) > 50


def test_the_path_after_the_boundary_does_change(worlds):
    cut = worlds["cutoff"]
    after = [(a, b) for a, b in zip(worlds["base"].decisions, worlds["pert"].decisions, strict=True) if pd.Timestamp(a.cutoff_utc) > cut]
    assert len(after) >= 5
    assert any(a.plan_hash != b.plan_hash for a, b in after)  # the rewritten future moves later decisions ...
    late = [k for k in worlds["base"].nav if k > worlds["t_date"]]
    assert any(worlds["base"].nav[k] != worlds["pert"].nav[k] for k in late)  # ... and later valuations: the control is not vacuous


def test_a_deliberate_look_ahead_would_be_detected(worlds, monkeypatch):
    """Negative control: let the perturbed world's decision at the boundary peek ten days ahead; it must stop matching the unperturbed world."""
    from quant_pipeline import engine as E

    ep, eq = worlds["base_p"].ev.engine, worlds["pert_p"].ev.engine
    params = _params(worlds)
    pos = worlds["dec"].position
    rpos = ep.refit_position(pos)
    honest, _ = ep.target(pos, params, ep.fit(rpos, params.n_components))
    real = E.agent_context
    monkeypatch.setattr(E, "agent_context", lambda cfg, md, cals, sess, fit, cut, **kw: real(cfg, md, cals, sess, fit, cut + pd.Timedelta(days=10), **kw))
    eq.caches.targets.clear()
    leaky, _ = eq.target(pos, params, eq.fit(rpos, params.n_components))
    assert not np.array_equal(honest.a_bar, leaky.a_bar)
    monkeypatch.undo()
    eq.caches.targets.clear()
    restored, _ = eq.target(pos, params, eq.fit(rpos, params.n_components))
    assert np.array_equal(honest.a_bar, restored.a_bar)  # and the honest path matches again


@pytest.fixture(scope="module")
def other_trials(worlds):
    """The remaining trials of the design (different PCA component counts, regime gates and objective weights), run on both worlds."""
    out = {}
    for t in worlds["base_p"].ev.design.trials[1:]:
        out[t["trial_hash"]] = (t, worlds["base_p"].ev.run_trial_path(t), worlds["pert_p"].ev.run_trial_path(t))
    return out


def test_every_other_trial_is_identical_up_to_the_boundary_and_differs_after_it(worlds, other_trials):
    """Look-ahead can hide in one trial's configuration only (another component count, gate or weight); each trial gets the same boundary test."""
    from quant_pipeline.trials import to_trial_params

    cut, t_date = worlds["cutoff"], worlds["t_date"]
    assert len(other_trials) == len(worlds["base_p"].ev.design.trials) - 1 >= 3
    shapes = {(to_trial_params(worlds["base_p"].cfg, t["parameters"]).n_components, t["parameters"]["regime.agent_gate_candidates"])
              for t, _, _ in other_trials.values()}
    assert len(shapes) == len(other_trials)  # the trials really are different configurations
    for h, (t, base, pert) in other_trials.items():
        before = [(a, b) for a, b in zip(base.decisions, pert.decisions, strict=True) if pd.Timestamp(a.cutoff_utc) <= cut]
        after = [(a, b) for a, b in zip(base.decisions, pert.decisions, strict=True) if pd.Timestamp(a.cutoff_utc) > cut]
        assert len(before) == DECISION_INDEX + 1 and len(after) >= 5, h
        for a, b in before:
            assert (a.position, a.cutoff_utc, a.plan_hash, a.a_bar, a.weights, a.w_minus, a.branch, a.n_orders, a.objective_terms) == \
                   (b.position, b.cutoff_utc, b.plan_hash, b.a_bar, b.weights, b.w_minus, b.branch, b.n_orders, b.objective_terms), (h, a.position)
        fb = [(f.order_id, f.ts, f.realized_quantity, f.status) for f in base.fills if pd.Timestamp(f.ts) <= cut]
        fq = [(f.order_id, f.ts, f.realized_quantity, f.status) for f in pert.fills if pd.Timestamp(f.ts) <= cut]
        assert fb == fq and len(fb) > 0, h
        assert {k: v for k, v in base.nav.items() if k <= t_date} == {k: v for k, v in pert.nav.items() if k <= t_date}, h
        assert any(a.plan_hash != b.plan_hash for a, b in after), h  # the rewritten future reaches this configuration too: not vacuous
