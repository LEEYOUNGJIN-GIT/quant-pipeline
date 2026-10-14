"""Nested evaluation (spec 11.1-11.8): inner trial selection, outer folds with continuous ledger carry, cost scenarios.

The holdout is never opened here. Committed outer paths run from the profile opening state, continue across folds with the exact
prior ledger and pending orders, and use only the hyperparameters frozen by their own selection pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from .benchmarks import make_policy, validate_benchmarks
from .calendars import Calendars, RowMap
from .config import Config
from .data import MarketData
from .decision import RegimeSource, TrialParams
from .engine import Caches, Opening, PathCarry, PathEngine, PathResult
from .hashing import hash_json_obj
from .splits import Block, InnerFold, SelectionPass, SplitManifest, purge_range_clash
from .trials import Selection, TrialDesign, TrialRegistry, enumerate_trials, select, to_trial_params

BENCHMARK_IDS = ("cash_krw", "equal_weight_periodic", "equal_weight_staged_entry", "equal_weight_buy_and_hold_reference")


@dataclass(frozen=True)
class InnerEvaluation:
    pass_name: str
    trial_hash: str
    inner_index: int
    block: str
    score: float
    eligible: bool
    ineligible_reason: str
    executable_decisions: int
    realized_fills: int
    scored_intervals: int
    turnover: float
    replay_hash: str
    opening_state_hash: str
    censored_decision: int


@dataclass(frozen=True)
class PassOutcome:
    name: str
    train_end: int
    evaluations: tuple[InnerEvaluation, ...]
    selection: Selection
    winner_hash: str
    winner_parameters: dict[str, Any]
    leave_one_fold_out_winners: tuple[str, ...]
    selection_unstable: bool


@dataclass
class Counts:
    trials: int = 0
    executed_paths: int = 0
    executed_decisions: int = 0
    no_sharing_replayed_decisions: int = 0
    trial_fold_evaluations: int = 0


@dataclass
class OuterCarry:
    strategy: PathCarry | None
    benchmarks: dict[str, PathCarry | None]
    extra: dict[str, Any]


@dataclass
class OuterRun:
    fold_id: str
    trial_hash: str
    multiplier: float
    result: PathResult
    benchmarks: dict[str, PathResult] = field(default_factory=dict)


def interval_returns(nav: dict[str, Decimal]) -> dict[str, float]:
    """Simple wealth relatives, keyed by the anchor session that ends each interval."""
    keys = list(nav)
    return {keys[i]: float(nav[keys[i]] / nav[keys[i - 1]] - 1) for i in range(1, len(keys))}


def net_excess_sharpe(returns: np.ndarray, rf: np.ndarray, periods_per_year: int) -> float:
    ex = np.asarray(returns, dtype=float) - np.asarray(rf, dtype=float)
    sd = ex.std(ddof=1) if len(ex) > 1 else 0.0
    return float(ex.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def complexity(design: TrialDesign, params: dict[str, Any], defaults: dict[str, Any]) -> float:
    return float(sum(1 for k in design.dimension_order if params[k] != defaults[k]))


class Evaluator:
    def __init__(self, cfg: Config, md: MarketData, cals: Calendars, sess: dict[str, pd.DatetimeIndex], rm: RowMap,
                 regime: RegimeSource, hashes: dict[str, str], manifest: SplitManifest, opening: Opening,
                 caches: Caches | None = None):
        validate_benchmarks(cfg)
        self.cfg, self.cals, self.rm, self.manifest, self.opening = cfg, cals, rm, manifest, opening
        self.engine = PathEngine(cfg, md, cals, sess, rm, regime, hashes, caches)
        self.design = enumerate_trials(cfg)
        self.defaults = {k: self.design.trials[0]["parameters"][k] for k in self.design.trials[0]["parameters"]}
        self.defaults = {name: fn(cfg)[0] for name, fn in __import__("quant_pipeline.trials", fromlist=["x"]).TRIAL_DIMENSIONS.items()}
        self.counts = Counts()
        self.registry = TrialRegistry("run", cfg.evaluation.max_parameter_trials)
        self._rf: dict[str, float] | None = None

    # ------------------------------------------------------------------------------------------ risk-free leg
    def risk_free(self) -> dict[str, float]:
        """Interval-matched accrued KRW cash return, from the cash reference ledger (spec 10.4)."""
        if self._rf is None:
            r = self.engine.run(label="cash", fold_id="rf", trial=TrialParams.first(self.cfg), trial_hash="benchmark:cash_krw",
                                multiplier=1.0, start=1, end=len(self.rm.eligible), opening=self.opening,
                                policy=make_policy(self.cfg, "cash_krw"))
            self._rf = interval_returns(r.nav)
        return self._rf

    # ------------------------------------------------------------------------------------------ inner selection
    def _check_prefix_sharing(self) -> None:
        for p in self.manifest.passes:
            for f in p.inner:
                if purge_range_clash(p.refit_positions, f):
                    raise NotImplementedError(f"{p.name}: a refit falls inside a purge range; endpoint paths are not prefix-identical")

    def run_trial_path(self, t: dict[str, Any]) -> PathResult:
        """One isolated path for one trial to the last validation endpoint (shared by every endpoint, see PROGRESS 5.5)."""
        last = max(self.manifest.path_endpoints())
        h = t["trial_hash"]
        label = f"inner:{h[7:19]}"
        r = self.engine.run(label=label, fold_id=label, trial=to_trial_params(self.cfg, t["parameters"]), trial_hash=h, multiplier=1.0,
                            start=1, end=last, opening=self.opening)
        self.counts.executed_paths += 1
        self.counts.executed_decisions += last
        return r

    def run_trial_paths(self) -> dict[str, PathResult]:
        self._check_prefix_sharing()
        out = {t["trial_hash"]: self.run_trial_path(t) for t in self.design.trials}
        self.counts.trials = len(out)
        return out

    def score_block(self, path: PathResult, block: Block, trial_hash: str, pass_name: str, inner: InnerFold) -> InnerEvaluation:
        rf = self.risk_free()
        rets = interval_returns(path.nav)
        sessions = [s for s in block.valuation_sessions if s in rets]
        r = np.array([rets[s] for s in sessions])
        f = np.array([rf[s] for s in sessions])
        ppy = self.cfg.evaluation.periods_per_year
        score = net_excess_sharpe(r, f, ppy) if len(r) else float("nan")
        ex_end = block.end_pos - 1  # the last decision's orders execute after the fence and are censored
        decs = [d for d in path.decisions if block.start_pos <= d.position <= ex_end]
        t_lo, t_hi = pd.Timestamp(block.first_decision_cutoff_utc), pd.Timestamp(block.fence_utc)
        fills = [x for x in path.fills if x.status in ("filled", "reduced_cash", "reduced_holdings")
                 and t_lo < pd.Timestamp(x.ts) <= t_hi]
        ev = self.cfg.evaluation.inner
        reasons = []
        if len(decs) < ev.minimum_decisions_per_validation:
            reasons.append("decisions")
        if len(fills) < ev.minimum_fills_per_validation:
            reasons.append("fills")
        if len(sessions) < ev.minimum_scored_nav_intervals_per_validation:
            reasons.append("intervals")
        if not np.isfinite(score):
            reasons.append("undefined_score")
        turnover = float(np.mean([d.one_way_turnover_intended for d in decs])) if decs else float("nan")
        replay = hash_json_obj([d.plan_hash for d in path.decisions if d.position <= inner.path_end])
        opening = hash_json_obj({"cash": str(self.opening.cash), "holdings": self.opening.holdings, "ts": str(self.opening.ts)})
        return InnerEvaluation(pass_name, trial_hash, inner.index, block.name, score, not reasons, ",".join(reasons), len(decs),
                               len(fills), len(sessions), turnover, replay, opening, inner.censored_decision)

    def selection_pass(self, p: SelectionPass, paths: dict[str, PathResult]) -> PassOutcome:
        evals: list[InnerEvaluation] = []
        for t in self.design.trials:
            for f in p.inner:
                e = self.score_block(paths[t["trial_hash"]], f.validation, t["trial_hash"], p.name, f)
                evals.append(e)
                self.counts.trial_fold_evaluations += 1
                self.counts.no_sharing_replayed_decisions += f.path_end
                self.registry.register(f"{p.name}:inner{f.index}", t["trial_hash"], "completed", score=e.score, eligible=e.eligible)
        by_trial: dict[str, list[InnerEvaluation]] = {}
        for e in evals:
            by_trial.setdefault(e.trial_hash, []).append(e)
        scores = {h: [e.score for e in es] for h, es in by_trial.items()}
        elig = {h: all(e.eligible for e in es) for h, es in by_trial.items()}
        turn = {h: float(np.mean([e.turnover for e in es])) for h, es in by_trial.items()}
        cx = {t["trial_hash"]: complexity(self.design, t["parameters"], self.defaults) for t in self.design.trials}
        sel = select(scores, turn, cx, self.cfg.evaluation.selection_score_tolerance, elig)
        loo = []
        for drop in range(len(p.inner)):
            keep = {h: [s for i, s in enumerate(v) if i != drop] for h, v in scores.items()}
            try:
                loo.append(select(keep, turn, cx, self.cfg.evaluation.selection_score_tolerance, elig).winner)
            except ValueError:
                loo.append("")
        params = next(t["parameters"] for t in self.design.trials if t["trial_hash"] == sel.winner)
        return PassOutcome(p.name, p.train_end, tuple(evals), sel, sel.winner, params, tuple(loo), len(set(loo)) > 1)

    def selection_from_paths(self, paths: dict[str, PathResult]) -> list[PassOutcome]:
        self.counts.trials = len(paths)
        return [self.selection_pass(p, paths) for p in self.manifest.passes]

    def run_selection(self) -> tuple[list[PassOutcome], dict[str, PathResult]]:
        paths = self.run_trial_paths()
        return self.selection_from_paths(paths), paths

    # ------------------------------------------------------------------------------------------ committed outer paths
    def run_outer_fold(self, fold, outcome: PassOutcome, multiplier: float, carry: OuterCarry | None,
                       with_benchmarks: bool = True) -> tuple[OuterRun, OuterCarry]:
        """One committed fold: starts from the exact end state of the previous fold and uses only its own frozen winner."""
        carry = carry or OuterCarry(None, dict.fromkeys(BENCHMARK_IDS), {})
        trial = to_trial_params(self.cfg, outcome.winner_parameters)
        r = self.engine.run(label=f"oos:{fold.fold_id}:x{multiplier:g}", fold_id=fold.fold_id, trial=trial, trial_hash=outcome.winner_hash,
                            multiplier=multiplier, start=fold.test.start_pos, end=fold.test.end_pos, opening=self.opening,
                            carry=carry.strategy)
        run = OuterRun(fold.fold_id, outcome.winner_hash, multiplier, r)
        benchmarks_carry = dict(carry.benchmarks)
        if with_benchmarks:
            for b in BENCHMARK_IDS:
                pol = make_policy(self.cfg, b)
                br = self.engine.run(label=f"{b}:{fold.fold_id}:x{multiplier:g}", fold_id=fold.fold_id, trial=trial,
                                     trial_hash=f"benchmark:{b}", multiplier=multiplier, start=fold.test.start_pos,
                                     end=fold.test.end_pos, opening=self.opening, carry=benchmarks_carry[b], policy=pol)
                benchmarks_carry[b] = br.carry
                run.benchmarks[b] = br
        return run, OuterCarry(r.carry, benchmarks_carry, {})

    def run_outer(self, outcomes: list[PassOutcome], multiplier: float = 1.0, with_benchmarks: bool = True) -> list[OuterRun]:
        """Continuous OOS ledger: fold k starts from the exact end state of fold k-1."""
        by_name = {o.name: o for o in outcomes}
        runs: list[OuterRun] = []
        carry: OuterCarry | None = None
        for fold in self.manifest.outer:
            run, carry = self.run_outer_fold(fold, by_name[fold.fold_id], multiplier, carry, with_benchmarks)
            runs.append(run)
        return runs

    def oos_returns(self, runs: list[OuterRun]) -> dict[str, np.ndarray]:
        """The continuous aligned outer-OOS valuation-return tuple (22 intervals on the fixture); holdout excluded."""
        nav: dict[str, Decimal] = {}
        for r in runs:
            nav.update(r.result.nav)
        rets = interval_returns(nav)
        sessions = [s for o in self.manifest.outer for s in o.test.valuation_sessions]
        out = {"strategy": np.array([rets[s] for s in sessions])}
        for b in BENCHMARK_IDS:
            bn: dict[str, Decimal] = {}
            for r in runs:
                if b in r.benchmarks:
                    bn.update(r.benchmarks[b].nav)
            if bn:
                br = interval_returns(bn)
                out[b] = np.array([br[s] for s in sessions])
        out["rf"] = np.array([self.risk_free()[s] for s in sessions])
        return out
