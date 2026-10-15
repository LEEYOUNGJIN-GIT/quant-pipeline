"""The path engine runs decisions at the scheduled decision rows only, not at every model row (spec 5.1, 6.1; INV-DATA-001).

Uses the deterministic SYNTHETIC research snapshot and a constant regime stub: this exercises the decision/fit/calibration plumbing, not
the HMM lane and not research performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline import engine as E
from quant_pipeline.calendars import Calendars, build_row_map, utc, window_from_manifest
from quant_pipeline.config import load_config
from quant_pipeline.data import load_initial_holdings, load_market_data
from quant_pipeline.decision import TrialParams
from quant_pipeline.fitstate import decision_cutoff, lineage_hashes
from quant_pipeline.snapshot import load_research_context
from support.synthetic_research import build_research_repo

CALS = Calendars.load(["XKRX", "XNYS"])


class StubRegime:
    def probs(self, cutoff):
        return np.array([0.7, 0.3]), ["low_risk", "high_risk"]


@pytest.fixture(scope="module")
def research(tmp_path_factory):
    root, _ = build_research_repo(tmp_path_factory.mktemp("engine"))
    cfg, snap = load_research_context(load_config(ROOT / "configs" / "research_8_12asset.yaml"), root)
    rm, sess = build_row_map(cfg, CALS, window_from_manifest(snap.manifest))
    md = load_market_data(cfg, root)
    cash, qty, asof = load_initial_holdings(cfg, root)
    eng = E.PathEngine(cfg, md, CALS, sess, rm, StubRegime(), lineage_hashes(cfg, root))
    return cfg, rm, eng, E.Opening(cash, qty, utc(asof))


def test_decision_cutoffs_follow_the_decision_session_not_the_model_row(research):
    _, rm, eng, _ = research
    assert eng.decision_positions == rm.decision_positions and len(eng.decision_positions) == 84 < len(rm.eligible)
    cfg = research[0]
    for d in rm.decision_rows:
        assert eng.decision_cutoffs[d.position] == decision_cutoff(CALS, cfg, d.session)
        model = eng.cutoffs[d.position - 1]
        assert (eng.decision_cutoffs[d.position] == model) if d.on_complete_case_row else (eng.decision_cutoffs[d.position] > model)
    assert eng.decisions_upto(0) == 0 and eng.decisions_upto(len(rm.eligible)) == 84 and eng.decisions_upto(22) == 3


def test_a_path_decides_only_at_scheduled_decisions_with_their_own_ids(research):
    cfg, rm, eng, opening = research
    end = eng.decision_positions[2] + 3  # three decisions plus three idle model rows
    res = eng.run(label="t", fold_id="f", trial=TrialParams.first(cfg, gate_name="balanced_2state"), trial_hash="sha256:x", multiplier=1.0,
                  start=1, end=end, opening=opening)
    assert [d.position for d in res.decisions] == list(eng.decision_positions[:3])
    assert [d.cutoff_utc for d in res.decisions] == [E.fmt(eng.decision_cutoffs[p]) for p in eng.decision_positions[:3]]
    ids = {d.decision_id for d in rm.decision_rows[:3]}
    assert all(o.order_id for p in res.plans for o in p.orders) and {p.decision_id for p in res.plans} == ids
    assert {f.decision_position for f in res.fills} <= set(eng.decision_positions[:3])
    assert [d.refit_position for d in res.decisions] == [eng.refit_position(p) for p in eng.decision_positions[:3]]
    assert eng.refit_position(1) == 1 and eng.refit_position(22) == 22 and eng.refit_position(21) == 1


def test_calibration_uses_only_decisions_inside_the_training_window(research, monkeypatch):
    cfg, _, eng, opening = research
    seen = []
    real = E.calibrate_fold

    def spy(*args, **kw):
        seen.append(len(args[7]))  # calibration_cutoffs
        return real(*args, **kw)

    monkeypatch.setattr(E, "calibrate_fold", spy)
    eng.caches.calibrations.clear()
    eng.run(label="t2", fold_id="f", trial=TrialParams.first(cfg, gate_name="balanced_2state"), trial_hash="sha256:x", multiplier=1.0,
            start=1, end=eng.decision_positions[2], opening=opening)
    fit_ends = [eng.cutoffs[r - 1] for r in (1, 22)]
    expected = [sum(1 for c in eng.decision_cutoffs.values() if c <= f) for f in fit_ends]
    assert seen == expected and seen[1] < 22  # decisions whose cutoff is inside the fit window, not one per model row


def test_the_fixture_engine_is_the_degenerate_case():
    from quant_pipeline.calendars import build_row_map as brm

    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    rm, sess = brm(cfg, CALS)
    md = load_market_data(cfg, ROOT)
    eng = E.PathEngine(cfg, md, CALS, sess, rm, StubRegime(), lineage_hashes(cfg, ROOT))
    assert eng.decision_positions == tuple(range(1, 71)) and list(eng.decision_cutoffs.values()) == eng.cutoffs
    assert [d.decision_id for d in rm.decision_rows][:2] == ["D001", "D002"] and eng.decisions_upto(17) == 17
