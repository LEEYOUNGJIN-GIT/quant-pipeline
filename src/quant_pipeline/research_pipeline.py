"""Resumable research pipeline (spec 12.1-12.3, 15): the fixture pipeline's stages on a verified immutable research snapshot.

Differences from the fixture lane: the snapshot is resolved, contamination-scanned and hydrated first; the row map and split manifest come
from the snapshot's collected window (decision rows are a subset of the model rows); the regime lane is the HMM-filtered one; the HMM fits
are persisted as the ``06_hmm_filtered_state`` stage; the holdout stays sealed. A SYNTHETIC snapshot (manifest ``synthetic: true``) is refused
unless the caller opts in explicitly, and then every summary and report context is labelled ``synthetic_engineering_test`` (OQ-046).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import serialization as S
from .calendars import build_row_map, window_from_manifest
from .checkpoints import CheckpointKey
from .config import Config, load_config
from .hmm_lane import HMMRegimeSource
from .pipeline import FixturePipeline
from .snapshot import ResolvedSnapshot, load_research_context

EVIDENCE_SYNTHETIC = "synthetic_engineering_test"
EVIDENCE_RESEARCH = "research_snapshot"


class SyntheticSnapshotRefused(RuntimeError):
    """A synthetic snapshot was supplied to a research run that did not explicitly allow it."""


class ResearchPipeline(FixturePipeline):
    regime_family = "hmm_filtered"

    def __init__(self, root: Path, out_root: Path, run_id: str, config: str = "configs/research_8_12asset.yaml",
                 stop_after: int | None = None, allow_synthetic: bool = False):
        self.allow_synthetic = allow_synthetic
        self.snapshot: ResolvedSnapshot | None = None
        super().__init__(root, out_root, run_id, config=config, stop_after=stop_after)

    # ------------------------------------------------------------------------------------------ hooks
    def _load_config(self, path: Path) -> Config:
        cfg = load_config(path)
        if cfg.data.mode != "snapshot" or cfg.regime.mode != "hmm_filtered":
            raise ValueError("the research pipeline needs a snapshot profile with regime.mode=hmm_filtered")
        cfg, self.snapshot = load_research_context(cfg, self.root)
        if self.snapshot.synthetic and not self.allow_synthetic:
            raise SyntheticSnapshotRefused(f"snapshot {self.snapshot.snapshot_id[:19]} is synthetic (an engineering test); it cannot feed a research run")
        return cfg

    @property
    def evidence_class(self) -> str:
        return EVIDENCE_SYNTHETIC if self.snapshot is not None and self.snapshot.synthetic else EVIDENCE_RESEARCH

    def _build_row_map(self):
        assert self.snapshot is not None
        return build_row_map(self.cfg, self.cals, window_from_manifest(self.snapshot.manifest))

    def _regime_source(self) -> HMMRegimeSource:
        self.hmm = HMMRegimeSource(self.cfg, self.md, self.cals, self.sess)
        return self.hmm

    def _extra_units(self, k_outer: list[CheckpointKey], k_sel: CheckpointKey) -> list[CheckpointKey]:
        k = self.key("06_hmm_filtered_state", "global")
        self.unit(k, [k_sel, *k_outer], lambda: {"hmm_fits.json": S.dumps(self._hmm_report())})
        return [k]

    # ------------------------------------------------------------------------------------------ evidence
    def _hmm_report(self) -> dict[str, Any]:
        fits = sorted((fr.report() for fr in self.hmm.fits.values()), key=lambda r: (r["key"], r["n_states"]))
        return {"evidence_class": self.evidence_class, "method": "hmmlearn EM -> export -> align -> owned forward filter (spec 7.5)",
                "seed_stream": "hmm_em", "seed": self.cfg.reproducibility.random_streams.hmm_em, "n_fits": len(fits), "fits": fits,
                "decisions_served": len(self.hmm.audit),
                "scopes": {s: sum(1 for a in self.hmm.audit if a["scope"] == s) for s in ("train", "boundary", "oos")},
                "all_converged": all(f["converged"] for f in fits)}

    def _summary(self, runs) -> dict[str, Any]:
        return {**super()._summary(runs), "evidence_class": self.evidence_class, "snapshot_id": None if self.snapshot is None else self.snapshot.snapshot_id}

    def _coverage(self, outcomes, outer, planned) -> dict[str, Any]:
        from collections import Counter

        m = self.manifest
        return {"evidence_class": self.evidence_class, "profile": self.cfg.profile.name, "snapshot_id": self.snapshot.snapshot_id,
                "model_positions": m.n_positions, "decision_rows": len(m.decision_positions), "outer_folds": len(m.outer),
                "holdout": {"positions": [m.holdout.start_pos, m.holdout.end_pos], "decisions": m.holdout.n_decisions, "opened": False},
                "trials": len(self.ev.design.trials), "trial_space": self.ev.design.cardinality,
                "checkpoint_stages": dict(Counter(k.stage for k in planned)),
                "scored_decisions_per_outer_fold": {o.fold_id: o.test.n_decisions for o in m.outer}}

    def projection(self, outcomes, outer, summary) -> dict[str, Any]:
        return {**super().projection(outcomes, outer, summary), "evidence_class": self.evidence_class}
