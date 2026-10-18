"""Research regime lane: per-fit EM, alignment and forward-only filtering (spec 7.5; INV-DATA-001, INV-HMM-001): synthetic data, offline."""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline import engine as E
from quant_pipeline import regimes
from quant_pipeline.calendars import Calendars, build_row_map, utc, window_from_manifest
from quant_pipeline.config import load_config
from quant_pipeline.data import MarketData, Series, load_initial_holdings, load_market_data
from quant_pipeline.decision import TrialParams
from quant_pipeline.fitstate import decision_cutoff, fit_core, hmm_features, lineage_hashes
from quant_pipeline.hmm_lane import HMMRegimeSource, RegimeHalt
from quant_pipeline.snapshot import load_research_context
from support.synthetic_research import build_research_repo

CALS = Calendars.load(["XKRX", "XNYS"])
FIT_POSITION = 300


def _oracle():
    spec = importlib.util.spec_from_file_location("hmm_probe_oracle", ROOT / "tools" / "hmm_probe_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root, _ = build_research_repo(tmp_path_factory.mktemp("hmm"))
    cfg, snap = load_research_context(load_config(ROOT / "configs" / "research_8_12asset.yaml"), root)
    rm, sess = build_row_map(cfg, CALS, window_from_manifest(snap.manifest))
    md = load_market_data(cfg, root)
    hashes = lineage_hashes(cfg, root)
    fit_end = decision_cutoff(CALS, cfg, rm.eligible[FIT_POSITION - 1])
    fit = fit_core(cfg, md, CALS, sess, fold_id="t", refit_id="r", fit_end_utc=fit_end, hashes=hashes)
    return {"root": root, "cfg": cfg, "rm": rm, "sess": sess, "md": md, "hashes": hashes, "fit": fit, "fit_end": fit_end}


def source(w, md=None) -> HMMRegimeSource:
    return HMMRegimeSource(w["cfg"], md or w["md"], CALS, w["sess"])


def trial(w, gate):
    return TrialParams.first(w["cfg"], gate_name=gate)


@pytest.mark.parametrize("gate,n", [("balanced_2state", 2), ("balanced_3state", 3)])
def test_filtered_probabilities_are_a_simplex_with_the_declared_labels(world, gate, n):
    src = source(world)
    bound = src.bind(world["fit"], trial(world, gate))
    p, labels = bound.probs(world["fit_end"])
    assert p.shape == (n,) and labels == world["cfg"].regime.state_labels_by_n_states[n]
    assert np.all(p >= 0) and abs(p.sum() - 1.0) < 1e-12 and labels[-1] == "high_risk"


def test_the_gate_decides_the_state_count(world):
    src = source(world)
    assert src.n_states_of(trial(world, "defensive_tilt_2state")) == 2 and src.n_states_of(trial(world, "defensive_tilt_3state")) == 3


def test_the_fit_is_deterministic_and_cached(world):
    a, b = source(world), source(world)
    pa = a.bind(world["fit"], trial(world, "balanced_2state")).probs(world["fit_end"])[0]
    pb = b.bind(world["fit"], trial(world, "balanced_2state")).probs(world["fit_end"])[0]
    assert np.array_equal(pa, pb)  # the EM seed is the declared hmm_em stream
    a.bind(world["fit"], trial(world, "defensive_tilt_2state"))  # same fit and state count: no second EM
    assert len(a.fits) == 1


def test_train_window_posteriors_match_the_independent_oracle(world):
    oracle = _oracle()
    w = world
    src = source(w)
    fr = src.fitted(w["fit"], 2)
    cc = w["fit"].panel[w["fit"].panel.complete_case]
    feats = hmm_features(w["fit"], cc[[f"{a}_return" for a in w["fit"].asset_ids]].to_numpy(float))
    s, a, m, v = fr.pipeline.aligned.arrays()
    _, filt = oracle.forward(s, a, m, v, feats)
    bound = src.bind(w["fit"], trial(w, "balanced_2state"))
    for i in (0, 10, len(filt) // 2, len(filt) - 1):
        assert np.max(np.abs(bound.probs(fr.train_cutoffs[i])[0] - filt[i])) < 1e-10


def test_a_decision_after_the_fit_continues_from_the_boundary_posterior_like_the_oracle(world):
    oracle = _oracle()
    w = world
    src = source(w)
    fr = src.fitted(w["fit"], 2)
    later = decision_cutoff(CALS, w["cfg"], w["rm"].eligible[FIT_POSITION + 9])  # ten model rows after the fit
    got = src.bind(w["fit"], trial(w, "balanced_2state")).probs(later)[0]
    assert later > w["fit_end"]
    d = w["cfg"].availability.price_publication_delay
    from quant_pipeline.calendars import parse_delay
    from quant_pipeline.panels import daily_panel
    xn = w["sess"]["XNYS"][[CALS.close("XNYS", s) + parse_delay(d) <= later for s in w["sess"]["XNYS"]]]
    panel = daily_panel(w["md"].total_return, CALS, xn, d)
    new = panel[panel.complete_case]
    new = new[pd.to_datetime(new.cutoff_utc, utc=True) > fr.train_cutoffs[-1]]
    assert len(new) == 10
    feats = hmm_features(w["fit"], new[[f"{a}_return" for a in w["fit"].asset_ids]].to_numpy(float))
    s, a, m, v = fr.pipeline.aligned.arrays()
    _, filt = oracle.forward(s, a, m, v, feats, prior=fr.pipeline.boundary_posterior)
    assert np.max(np.abs(got - filt[-1])) < 1e-10


def _perturbed(md: MarketData, after: pd.Timestamp) -> MarketData:
    """Total-return levels observed after ``after`` are scaled; everything available by ``after`` is untouched."""
    cut = np.datetime64(after.tz_convert("UTC").tz_localize(None))
    tr = {}
    for a, s in md.total_return.items():
        scale = np.where(s.available > cut, 1.6, 1.0)
        tr[a] = Series(s.event, s.available, s.value * scale, s.quality)
    return dataclasses.replace(md, total_return=tr)


def test_data_after_a_decision_cutoff_cannot_change_its_posterior(world):
    w = world
    c1 = decision_cutoff(CALS, w["cfg"], w["rm"].eligible[FIT_POSITION + 4])
    c2 = decision_cutoff(CALS, w["cfg"], w["rm"].eligible[FIT_POSITION + 5])  # the first row after the rewritten future
    base, mutated = source(w), source(w, _perturbed(w["md"], c1))
    t = trial(w, "balanced_2state")
    p1 = base.bind(w["fit"], t).probs(c1)[0]
    assert np.array_equal(p1, mutated.bind(w["fit"], t).probs(c1)[0])  # the future was rewritten; the past decision did not move
    assert not np.allclose(base.bind(w["fit"], t).probs(c2)[0], mutated.bind(w["fit"], t).probs(c2)[0])  # the perturbation is real


def test_a_non_converged_em_fit_halts_instead_of_producing_probabilities(world, monkeypatch):
    real = regimes.fit_em

    def not_converged(*a, **k):
        return dataclasses.replace(real(*a, **k), converged=False)

    monkeypatch.setattr(regimes, "fit_em", not_converged)
    with pytest.raises(RegimeHalt, match="did not converge"):
        source(world).bind(world["fit"], trial(world, "balanced_2state"))


def test_a_decision_on_an_anchor_without_a_new_model_row_uses_the_last_filtered_posterior(world):
    w = world
    rm = w["rm"]
    after = [s for s in rm.anchors if s > rm.eligible[FIT_POSITION - 1] and s not in set(rm.complete_case)]
    assert after, "the synthetic calendar has an XKRX-only holiday after the fit"
    anchor = after[0]
    prev = max(s for s in rm.complete_case if s < anchor)
    src = source(w)
    bound = src.bind(w["fit"], trial(w, "balanced_2state"))
    held = bound.probs(decision_cutoff(CALS, w["cfg"], anchor))[0]
    last = bound.probs(decision_cutoff(CALS, w["cfg"], prev))[0]
    assert np.array_equal(held, last)
    held_row, last_row = src.audit[-2], src.audit[-1]  # the held decision and the decision on the last model row
    assert held_row["observed_cutoff"] == last_row["observed_cutoff"] and held_row["cutoff"] != held_row["observed_cutoff"]


@pytest.mark.parametrize("gate,n", [("balanced_2state", 2), ("balanced_3state", 3)])
def test_the_engine_decides_with_the_hmm_lane(world, gate, n):
    w = world
    cash, qty, asof = load_initial_holdings(w["cfg"], w["root"])
    eng = E.PathEngine(w["cfg"], w["md"], CALS, w["sess"], w["rm"], source(w), w["hashes"])
    res = eng.run(label="h", fold_id="f", trial=trial(w, gate), trial_hash="sha256:h", multiplier=1.0, start=1,
                  end=eng.decision_positions[2], opening=E.Opening(cash, qty, utc(asof)))
    assert len(res.decisions) == 3
    tgt, _ = eng.target(eng.decision_positions[2], trial(w, gate), eng.fit(eng.refit_position(eng.decision_positions[2])))
    assert len(tgt.regime_probabilities) == n and abs(sum(tgt.regime_probabilities) - 1) < 1e-12
