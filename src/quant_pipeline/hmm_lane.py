"""Research regime lane (spec 7.5, 8.3; ``regime.mode: hmm_filtered``): per fit, EM -> export -> align -> owned forward filter.

Investable probabilities are the filtered posteriors ``p(z_t | y_1..y_t)`` only; there is no backward pass. For a decision inside a
fit's training window the probability is the train-window filter at the last model row with ``cutoff <= decision cutoff``; for a later
decision the filter continues from the boundary posterior over the model rows observed since the fit. A decision on an anchor without
a new model row (the other market is closed) uses the last filtered posterior and records its age (OQ-045). An EM fit that does not
converge halts the run (OQ-044).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .calendars import Calendars, parse_delay, utc
from .config import Config
from .data import MarketData
from .fitstate import HMM_FEATURE_ORDER, FoldFit, hmm_features
from .panels import daily_panel
from .regimes import HMMPipelineResult, forward_filter, run_hmm_pipeline

if TYPE_CHECKING:
    from .decision import TrialParams


class RegimeHalt(RuntimeError):
    """The regime layer cannot supply a trustworthy investable probability (non-convergence, no observation, degenerate fit)."""


@dataclass(frozen=True)
class FittedRegime:
    key: str
    n_states: int
    labels: tuple[str, ...]
    pipeline: HMMPipelineResult
    train_cutoffs: tuple[pd.Timestamp, ...]  # cutoffs of the complete-case rows the model was fitted and filtered on
    fit: FoldFit

    def report(self) -> dict:
        raw = self.pipeline.raw
        return {"key": self.key, "n_states": self.n_states, "labels": list(self.labels), "converged": raw.converged, "fitter": raw.fitter,
                "permutation": list(self.pipeline.permutation), "start_fallback_stationary": self.pipeline.start_fallback_stationary,
                "n_train_rows": len(self.train_cutoffs), "boundary_posterior": self.pipeline.boundary_posterior.tolist(),
                "aligned_parameters": self.pipeline.aligned.to_json()}


@dataclass
class HMMRegimeSource:
    """Factory of per-fit regime sources. ``bind(fit, trial)`` returns an object with ``probs(cutoff)`` (the decision's RegimeSource)."""

    cfg: Config
    md: MarketData
    cals: Calendars
    sess: dict[str, pd.DatetimeIndex]
    fits: dict[tuple[str, int], FittedRegime] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)

    def n_states_of(self, trial: TrialParams) -> int:
        gate = next(g for g in self.cfg.regime.agent_gate_candidates if g.name == trial.gate_name)
        return gate.applies_to_n_states

    def fitted(self, fit: FoldFit, n_states: int) -> FittedRegime:
        key = (fit.key, n_states)
        if key not in self.fits:
            self.fits[key] = self._fit(fit, n_states)
        return self.fits[key]

    def _fit(self, fit: FoldFit, n_states: int) -> FittedRegime:
        cols = [f"{a}_return" for a in fit.asset_ids]
        cc = fit.panel[fit.panel.complete_case]
        feats = hmm_features(fit, cc[cols].to_numpy(float))
        h = self.cfg.regime.hmm
        try:
            pipe = run_hmm_pipeline(feats, None, n_states=n_states, seed=self.cfg.reproducibility.random_streams.hmm_em,
                                    max_iter=h.max_iterations, tol=h.em_tolerance, feature_order=HMM_FEATURE_ORDER)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            raise RegimeHalt(f"HMM fit failed for {fit.refit_id} with {n_states} states: {e}") from e
        if not pipe.raw.converged:
            raise RegimeHalt(f"HMM EM did not converge for {fit.refit_id} with {n_states} states (max_iterations={h.max_iterations}); "
                             "a non-converged model is not investable")
        labels = tuple(self.cfg.regime.state_labels_by_n_states[n_states])
        return FittedRegime(fit.key, n_states, labels, pipe, tuple(utc(c) for c in cc.cutoff_utc), fit)

    def bind(self, fit: FoldFit, trial: TrialParams) -> BoundRegime:
        return BoundRegime(self, self.fitted(fit, self.n_states_of(trial)))

    def probs_at(self, fr: FittedRegime, cutoff: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
        cut = utc(cutoff)
        cuts = fr.train_cutoffs
        if cut <= cuts[-1]:
            i = int(np.searchsorted(np.array([c.value for c in cuts]), cut.value, side="right")) - 1
            if i < 0:
                raise RegimeHalt(f"no filtered regime is available by {cut}")
            self.audit.append({"fit": fr.key, "cutoff": str(cut), "observed_cutoff": str(cuts[i]), "scope": "train"})
            return fr.pipeline.train.filtered[i].copy(), list(fr.labels)
        d = parse_delay(self.cfg.availability.price_publication_delay)
        xnys = self.sess["XNYS"][[self.cals.close("XNYS", s) + d <= cut for s in self.sess["XNYS"]]]
        panel = daily_panel(self.md.total_return, self.cals, xnys, self.cfg.availability.price_publication_delay)
        cc = panel[panel.complete_case]
        new = cc[pd.to_datetime(cc.cutoff_utc, utc=True) > cuts[-1]]
        if new.empty:  # no model row since the fit: the boundary posterior is the latest filtered one
            self.audit.append({"fit": fr.key, "cutoff": str(cut), "observed_cutoff": str(cuts[-1]), "scope": "boundary"})
            return fr.pipeline.boundary_posterior.copy(), list(fr.labels)
        cols = [f"{a}_return" for a in fr.fit.asset_ids]
        oos = forward_filter(fr.pipeline.aligned, hmm_features(fr.fit, new[cols].to_numpy(float)), prior=fr.pipeline.boundary_posterior,
                             first_is_start=False)
        self.audit.append({"fit": fr.key, "cutoff": str(cut), "observed_cutoff": str(utc(new.cutoff_utc.iloc[-1])), "scope": "oos"})
        return oos.filtered[-1].copy(), list(fr.labels)


@dataclass(frozen=True)
class BoundRegime:
    source: HMMRegimeSource
    fitted: FittedRegime

    def probs(self, cutoff: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
        return self.source.probs_at(self.fitted, cutoff)
