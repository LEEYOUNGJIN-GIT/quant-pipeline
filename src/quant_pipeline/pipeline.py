"""Resumable fixture pipeline (spec 12.1-12.3): every expensive unit is checkpointed, resume follows the exact dependency chain,
and a fixture-only hook can stop the run immediately after a declared completed checkpoint.

Run as ``python -m quant_pipeline.pipeline --run-id ID --out-root DIR [--stop-after N] [--projection-out PATH]``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import serialization as S
from .bootstrap import run_bootstrap
from .calendars import Calendars, build_row_map, utc
from .checkpoints import CheckpointKey, CheckpointStore
from .config import load_config
from .data import load_initial_holdings, load_market_data
from .decision import FixedRegimeSource
from .dependency import accepted_lock_hash
from .engine import Opening
from .evaluation import BENCHMARK_IDS, Evaluator, OuterCarry
from .fitstate import lineage_hashes
from .hashing import hash_json_obj, hash_text
from .intervals import interval_report
from .regimes import FixedRegimeTable
from .reporting import build_context, render, scan_terminology
from .splits import build_split_manifest
from .trials import enumerate_trials, to_trial_params

MULTIPLIERS = (1.0, 0.0, 2.0)


class StopForTest(RuntimeError):
    """Fixture-only hook: raised immediately after a declared checkpoint has been durably written."""


@dataclass
class Progress:
    path: Path
    lines: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        self.lines.append(text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")


class FixturePipeline:
    def __init__(self, root: Path, out_root: Path, run_id: str, config: str = "configs/fixture_3asset.yaml", stop_after: int | None = None):
        self.root, self.run_id, self.stop_after = Path(root), run_id, stop_after
        self.cfg = load_config(self.root / config)
        self.store = CheckpointStore(out_root)
        self.cals = Calendars.load(["XKRX", "XNYS"])
        self.rm, self.sess = build_row_map(self.cfg, self.cals)
        self.md = load_market_data(self.cfg, self.root)
        hashes = lineage_hashes(self.cfg, self.root)
        self.hashes = hashes
        self.lineage = {"spec_hash": hashes["spec_hash"], "config_hash": hashes["base_profile_hash"],
                        "data_manifest_hash": hashes["data_manifest_hash"], "lock_hash": accepted_lock_hash(self.root),
                        "code_hash": hashes["code_hash"]}
        cash, qty, asof = load_initial_holdings(self.cfg, self.root)
        self.opening = Opening(cash, qty, utc(asof))
        self.manifest = build_split_manifest(self.cfg, self.cals, self.rm)
        regime = FixedRegimeSource(FixedRegimeTable.load(self.root / self.cfg.regime.fixed_table.path))
        self.ev = Evaluator(self.cfg, self.md, self.cals, self.sess, self.rm, regime, hashes, self.manifest, self.opening)
        self.progress = Progress(out_root / run_id / "progress_log.txt")
        self.written = 0
        self.outcomes = None
        self.outer_runs: dict = {}
        self.final_carry: dict = {}
        self.summary: dict = {}
        self.computed: list[str] = []
        self.loaded: list[str] = []

    # ------------------------------------------------------------------------------------------ unit machinery
    def key(self, stage: str, fold: str, trial: str = "none", refit: str = "none") -> CheckpointKey:
        return CheckpointKey(self.run_id, fold, trial, stage, refit)

    def unit(self, key: CheckpointKey, parents: list[CheckpointKey], compute, meta: dict[str, Any] | None = None) -> dict[str, bytes]:
        """Load a valid checkpoint, or compute and write it. Invalid leftovers are quarantined, never reused."""
        why = self.store.discard_invalid(key, self.lineage)
        if why:
            self.progress.note(f"quarantined invalid checkpoint {key.ident()}: {why}")
        ok, _ = self.store.validate(key, self.lineage)
        if ok:
            d = self.store.path(key)
            self.loaded.append(key.ident())
            return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file() and p.name not in ("COMPLETE", "checkpoint.json")}
        artifacts = compute()
        self.store.write(key, artifacts, lineage=self.lineage, parents=parents, meta=meta)
        self.computed.append(key.ident())
        self.written += 1
        self.progress.note(f"wrote checkpoint {key.ident()}")
        if self.stop_after is not None and self.written >= self.stop_after:
            raise StopForTest(f"stopped after {self.written} checkpoints (fixture test hook)")
        return artifacts

    # ------------------------------------------------------------------------------------------ the run
    def run(self) -> dict[str, Any]:
        self.progress.note(f"start run {self.run_id}; lineage {hash_json_obj(self.lineage)[:19]}")
        design = enumerate_trials(self.cfg)
        k_cfg = self.key("01_config_and_manifest", "global")
        self.unit(k_cfg, [], lambda: {"manifest.json": S.dumps({"split_manifest_hash": self.manifest.hash, "design_hash": design.design_hash,
                                                              "trials": [t["trial_hash"] for t in design.trials]}),
                                           "oos_intervals.json": S.dumps(interval_report(self.cfg, self.cals, self.rm, self.manifest))})

        paths, k_paths = {}, []
        for t in design.trials:
            k = self.key("10_event_time_execution_and_accounting", "selection", t["trial_hash"][7:19])
            k_paths.append(k)
            art = self.unit(k, [k_cfg], lambda t=t: S.path_to_artifacts(self.ev.run_trial_path(t)), {"trial_hash": t["trial_hash"]})
            paths[t["trial_hash"]] = S.path_from_artifacts(art)

        k_sel = self.key("11_oos_evaluation", "selection")
        art = self.unit(k_sel, k_paths, lambda: {"outcomes.json": S.dumps([S.outcome_to_obj(o) for o in self.ev.selection_from_paths(paths)])})
        outcomes = [S.outcome_from_obj(o) for o in S.loads(art["outcomes.json"])]
        by_name = {o.name: o for o in outcomes}

        outer: dict[float, list] = {}
        k_outer = []
        final_carry: dict[float, Any] = {}
        for mult in MULTIPLIERS:
            carry: OuterCarry | None = None
            prev: CheckpointKey | None = None
            outer[mult] = []
            for fold in self.manifest.outer:
                win = by_name[fold.fold_id]
                k = self.key("11_oos_evaluation", f"{fold.fold_id}-x{mult:g}", f"selected-{win.winner_hash[7:19]}")
                holder: dict[str, Any] = {}

                def compute(fold=fold, win=win, mult=mult, carry=carry, holder=holder):
                    run, new_carry = self.ev.run_outer_fold(fold, win, mult, carry)
                    holder["pair"] = (run, new_carry)
                    return S.outer_to_artifacts(run, new_carry)

                art = self.unit(k, [k_sel] + ([prev] if prev else []), compute, {"winner": win.winner_hash, "multiplier": mult})
                run, carry = S.outer_from_artifacts(art)
                outer[mult].append(run)
                k_outer.append(k)
                prev = k
            final_carry[mult] = carry

        self.outcomes, self.outer_runs, self.final_carry = outcomes, outer, final_carry
        k_final = self.key("11_oos_evaluation", "final", "freeze")
        art = self.unit(k_final, [k_sel], lambda: {"final_selection.json": S.dumps(self._final_selection(by_name["final"]))},
                        {"winner": by_name["final"].winner_hash})
        k_diag = self.key("13_verification", "global", "agent-diagnostics")
        self.unit(k_diag, [k_final], lambda: {"agent_pairwise_diagnostics.json": S.dumps(self._agent_diagnostics(by_name["final"]))})
        k_sum = self.key("12_bootstrap_and_reports", "global")
        art = self.unit(k_sum, k_outer, lambda: {"summary.json": S.dumps(self._summary(outer[1.0]))})
        self.summary = S.loads(art["summary.json"])
        k_cov = self.key("13_verification", "global", "coverage")
        planned = [k_cfg, *k_paths, k_sel, *k_outer, k_final, k_diag, k_sum, k_cov, self.key("12_bootstrap_and_reports", "global", "pre-holdout")]
        self.unit(k_cov, [k_sum, k_diag], lambda: {"fixture_coverage.json": S.dumps(self._coverage(outcomes, outer, planned))})
        k_rep = self.key("12_bootstrap_and_reports", "global", "pre-holdout")
        self.unit(k_rep, [k_sum, k_final], lambda: self._pre_holdout_report(outcomes, outer, self.summary))
        self.progress.note("run complete")
        return self.projection(outcomes, outer, self.summary)

    def _coverage(self, outcomes, outer, planned) -> dict[str, Any]:
        from collections import Counter

        from .coverage import coverage_matrix

        eng = self.ev.engine
        used = tuple(sorted({eng.refit_position(pos) for pos in range(1, self.manifest.n_positions + 1)}))
        lock = json.loads((self.root / self.cfg.data.manifest).parent.joinpath("fixture_generation.lock.json").read_text(encoding="utf-8"))
        return coverage_matrix(manifest=self.manifest, outcomes=outcomes, outer=outer, summary=self.summary, refit_positions=self.manifest.refit_positions,
                               refits_used=used, checkpoint_stages=dict(Counter(k.stage for k in planned)),
                               generation_lock={k: lock[k] for k in ("scenario_id", "generator_source_hash", "scenario_definition_hash", "calendar_session_digest")})

    def _agent_diagnostics(self, final) -> dict[str, Any]:
        """Pairwise agent diagnostics (spec 8.2) at the frozen candidate, for the calibration and outer-OOS scopes; the holdout scope
        is produced only when the holdout is opened."""
        from .diagnostics import scope_report

        scopes = {"calibration": list(range(1, self.manifest.passes[-1].train_end + 1)),
                  "outer_oos": [p for o in self.manifest.outer for p in o.test.decision_positions]}
        return scope_report(self.ev.engine, scopes, to_trial_params(self.cfg, final.winner_parameters))

    def _final_selection(self, final) -> dict[str, Any]:
        """Everything the holdout is allowed to use, frozen before it is touched (spec 11.6)."""
        cfg = self.cfg
        gate = cfg.regime.agent_gate_candidates[0]
        return {
            "frozen": True, "holdout_sealed": True, "created_before_holdout_access": True,
            "model_family": ["Trend", "MeanReversion", "Defensive", "Diversification", "PCA", "LedoitWolf", "fixed_regime_table"],
            "trial_hash": final.winner_hash, "parameters": final.winner_parameters, "ranking": list(final.selection.ranking),
            "selection_margin": final.selection.margin, "tie_path": list(final.selection.tie_path),
            "gate_table": {"name": gate.name, "family": gate.family, "by_state_label": gate.by_state_label},
            "threshold_policy": cfg.risk_pressure_calibration.model_dump(mode="json"),
            "objective_rules": cfg.objective.model_dump(mode="json"), "costs": cfg.costs.model_dump(mode="json"),
            "decision_schedule": cfg.decision_schedule.model_dump(mode="json"),
            "universe": [a.asset_id for a in (cfg.universe.assets or [])], "refit_policy": cfg.final_holdout.refit_policy,
            "split_manifest_hash": self.manifest.hash, "design_hash": self.ev.design.design_hash,
            "pre_holdout_information_fence_utc": self.manifest.outer[-1].test.fence_utc,
            "holdout_geometry": {"positions": [self.manifest.holdout.start_pos, self.manifest.holdout.end_pos],
                                 "valuation_sessions": list(self.manifest.holdout.valuation_sessions)},
            "lineage": self.lineage}

    def _pre_holdout_report(self, outcomes, outer, summary) -> dict[str, bytes]:
        ctx = build_context(self.cfg, self.ev, outcomes, outer, summary, self.lineage, self.run_id)
        md, html = render(ctx)
        scan = scan_terminology(md + "\n" + html)
        if scan["forbidden"] or scan["missing"]:
            raise ValueError(f"report failed the terminology scan: {scan}")
        return {"report.md": md.encode("utf-8"), "report.html": html.encode("utf-8"), "terminology_scan.json": S.dumps(scan),
                "report_context.json": S.dumps({k: v for k, v in ctx.items() if k != "lineage"})}

    def _summary(self, runs) -> dict[str, Any]:
        ret = self.ev.oos_returns(runs)
        b = self.cfg.bootstrap
        res = run_bootstrap({k: v for k, v in ret.items() if k != "rf"}, ret["rf"], replicates=b.replicates, mean_block_length=b.mean_block_length,
                            seed=self.cfg.reproducibility.random_streams.bootstrap, confidence=b.confidence_level,
                            periods_per_year=self.cfg.evaluation.periods_per_year, minimum_rows=b.minimum_oos_rows,
                            expected_rows=b.expected_oos_rows, inferential=b.intervals_are_inferential)
        return {"oos_intervals": {k: len(v) for k, v in ret.items()}, "bootstrap_status": res.status, "bootstrap_index_hash": res.index_hash,
                "bootstrap_shape": list(res.indices.shape), "statistics": res.statistics,
                "returns": {k: [float(x) for x in v] for k, v in ret.items()}}

    # ------------------------------------------------------------------------------------------ comparable output
    def projection(self, outcomes, outer, summary) -> dict[str, Any]:
        """The canonical output projection compared between a clean and a resumed lineage (no volatile fields)."""
        reg = {f"{o.name}|{e.trial_hash}|{e.inner_index}": [None if e.score != e.score else round(e.score, 12), e.eligible] for o in outcomes
               for e in o.evaluations}
        paths = {}
        for mult, runs in outer.items():
            for r in runs:
                paths[f"{r.fold_id}|x{mult:g}"] = {
                    "plans": [d.plan_hash for d in r.result.decisions], "nav": {k: str(v) for k, v in r.result.nav.items()},
                    "ledger_hash": hash_text(r.result.ledger_rows.to_csv(index=False, lineterminator="\n").encode("utf-8")),
                    "costs": str(r.result.costs), "fills": [(f.order_id, f.ts, f.realized_quantity, f.status) for f in r.result.fills],
                    "benchmarks": {b: hash_json_obj({k: str(v) for k, v in br.nav.items()}) for b, br in r.benchmarks.items()}}
        return {"winners": {o.name: o.winner_hash for o in outcomes}, "registry": reg, "paths": paths,
                "bootstrap_index_hash": summary["bootstrap_index_hash"], "oos_intervals": summary["oos_intervals"],
                "split_manifest_hash": self.manifest.hash, "lineage": self.lineage}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--config", default=None, help="profile path relative to --root (default: the pipeline default)")
    ap.add_argument("--projection-out", type=Path, default=None)
    args = ap.parse_args()
    p = FixturePipeline(args.root.resolve(), args.out_root, args.run_id, stop_after=args.stop_after,
                        **({"config": args.config} if args.config else {}))
    try:
        proj = p.run()
    except StopForTest as e:
        print(f"STOPPED: {e}")
        return 75
    if args.projection_out:
        args.projection_out.write_text(json.dumps(proj, sort_keys=True), encoding="utf-8")
    print(f"complete: computed={len(p.computed)} loaded={len(p.loaded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BENCHMARK_IDS", "FixturePipeline", "StopForTest", "np"]
