"""The cached full-history view must give every decision exactly the context the original per-decision panel rebuild gave (bit for bit)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.calendars import parse_delay, research_warmup_rows, utc
from quant_pipeline.fitstate import agent_context
from quant_pipeline.panels import daily_panel
from quant_pipeline.pipeline import FixturePipeline

ROOT = Path(__file__).resolve().parents[2]


def legacy_context_inputs(cfg, md, cals, sess, fit, cut, calibration):
    """The pre-optimisation construction: rebuild the whole panel up to ``cut`` from scratch."""
    assets = fit.asset_ids
    d = parse_delay(cfg.availability.price_publication_delay)
    cut = utc(cut)
    xnys = sess["XNYS"][[cals.close("XNYS", s) + d <= cut for s in sess["XNYS"]]]
    panel = daily_panel(md.total_return, cals, xnys, cfg.availability.price_publication_delay)
    cc = panel[panel.complete_case]
    returns = cc[[f"{a}_return" for a in assets]].to_numpy(float)
    n_fit = int((pd.to_datetime(cc.cutoff_utc, utc=True) <= fit.fit_end_utc).sum())
    illiquid = np.array([1.0 - np.mean([md.tradable(a, utc(c)) for a in assets]) for c in cc.cutoff_utc])
    warm = cfg.data.fixture_window.warmup_rows if cfg.data.fixture_window else research_warmup_rows(cfg)
    return returns, illiquid, tuple(cc.cutoff_utc), warm, (len(returns) - 1) if calibration else n_fit - 1


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    return FixturePipeline(ROOT, tmp_path_factory.mktemp("dv") / "out", "dv")


def test_decision_and_calibration_contexts_match_the_per_decision_rebuild_exactly(pipe):
    eng = pipe.ev.engine
    checked = 0
    for pos, cut in list(eng.decision_cutoffs.items()):
        fit = eng.fit(eng.refit_position(pos), None)
        for calibration in (False, True):
            if (cut >= fit.fit_end_utc) if not calibration else (cut <= fit.fit_end_utc):
                ctx = agent_context(pipe.cfg, pipe.md, pipe.cals, pipe.sess, fit, cut, calibration=calibration)
                returns, illiquid, row_cutoffs, warm, train_end = legacy_context_inputs(pipe.cfg, pipe.md, pipe.cals, pipe.sess, fit, cut, calibration)
                assert np.array_equal(ctx.returns, returns) and ctx.returns.dtype == returns.dtype
                # the memory layout feeds BLAS and decides the summation order: equal values in another layout move results by one ulp
                assert ctx.returns.strides == returns.strides and ctx.returns.flags.f_contiguous == returns.flags.f_contiguous
                assert np.array_equal(ctx.illiquid_history, illiquid) and ctx.row_cutoffs == row_cutoffs
                assert (ctx.train_start_row, ctx.train_end_row) == (warm, train_end)
                checked += 1
    assert checked >= 20


def test_a_caller_cannot_corrupt_the_cached_history_through_a_context(pipe):
    eng = pipe.ev.engine
    pos, cut = next(iter(eng.decision_cutoffs.items()))
    last = list(eng.decision_cutoffs.items())[-1]
    fit = eng.fit(eng.refit_position(last[0]), None)
    a = agent_context(pipe.cfg, pipe.md, pipe.cals, pipe.sess, fit, last[1])
    if a.returns.flags.writeable:
        a.returns[:] = 123.0
    else:
        with pytest.raises(ValueError):
            a.returns[:] = 123.0  # read-only: the cached history cannot be altered through the context
    a.illiquid_history[:] = 9.0
    b = agent_context(pipe.cfg, pipe.md, pipe.cals, pipe.sess, fit, last[1])
    assert not np.any(b.returns == 123.0) and not np.any(b.illiquid_history == 9.0)
    del pos, cut
