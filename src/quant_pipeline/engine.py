"""Sequential path engine (spec 6, 9, 11.1): decision -> target -> optimize -> plan -> global event queue -> ledger -> valuation.

One engine run is one isolated ledger path. Shared, trial-independent work (core fits, calibrations, targets) is cached under
dependency fingerprints (spec 3.4); anything that depends on a trial's optimizer or ledger is never shared.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from .benchmarks import BenchmarkPolicy, DecisionInput
from .calendars import Calendars, RowMap, fmt, parse_delay, utc
from .config import Config
from .data import MarketData, NoDataError
from .decision import DecisionTarget, RegimeSource, TrialParams, calibrate_fold, decide
from .dependency import build_fingerprint, cache_key, default_numerical_options
from .event_time import Event, build_events
from .fitstate import FoldFit, agent_context, decision_cutoff, fit_core
from .fx import krw_value
from .hashing import hash_json_obj
from .objective import ObjectiveCalibration, expected_cost_rates
from .optimizer import OptimizationResult, OptimizerInputs, optimize
from .rebalance import LedgerSnapshot, RebalancePlan, build_plan, mark_state
from .replay import LedgerRunner, RunnerState


class EngineHalt(RuntimeError):
    """A baseline invariant could not be upheld; the ledger up to this point is preserved by the caller."""


@dataclass(frozen=True)
class Opening:
    cash: Decimal
    holdings: dict[str, int]
    ts: pd.Timestamp


@dataclass
class Caches:
    fits: dict[Any, FoldFit] = field(default_factory=dict)
    calibrations: dict[str, ObjectiveCalibration] = field(default_factory=dict)
    targets: dict[tuple, DecisionTarget] = field(default_factory=dict)
    agent_memo: dict = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: {"fits": 0, "calibrations": 0, "targets": 0, "cache_hits": 0})


@dataclass
class FillRecord:
    order_id: str
    decision_position: int
    asset_id: str
    ts: str
    planned_quantity: int
    realized_quantity: int
    status: str
    origin: dict[str, str]


@dataclass
class DecisionRecord:
    position: int
    cutoff_utc: str
    refit_position: int
    a_bar: list[float]
    weights: list[float]
    w_minus: list[float]
    b_final: float
    branch: str
    n_attempts: int
    plan_hash: str
    n_orders: int
    shortfall_orders: list[str]
    one_way_turnover_intended: float
    pending_at_decision: int
    censored: bool
    objective_terms: tuple[float, float, float] | None


@dataclass
class PathCarry:
    runner: RunnerState
    pending: list[Event]
    drained_ts: pd.Timestamp
    last_weights: np.ndarray | None
    policy_state: dict | None = None


@dataclass
class PathResult:
    label: str
    decisions: list[DecisionRecord]
    fills: list[FillRecord]
    nav: dict[str, Decimal]  # valuation anchor session -> NAV (KRW)
    ledger_rows: pd.DataFrame
    carry: PathCarry
    reconciliation: list[dict]
    plans: list[RebalancePlan]
    optimizer_attempts: int
    costs: Decimal


class PathEngine:
    def __init__(self, cfg: Config, md: MarketData, cals: Calendars, sess: dict[str, pd.DatetimeIndex], rm: RowMap,
                 regime: RegimeSource, hashes: dict[str, str], caches: Caches | None = None):
        self.cfg, self.md, self.cals, self.sess, self.rm, self.regime, self.hashes = cfg, md, cals, sess, rm, regime, hashes
        self.caches = caches or Caches()
        self.static = build_events(cfg, cals, rm, md.actions, md.rates)
        self.cutoffs = [decision_cutoff(cals, cfg, s) for s in rm.eligible]  # one per MODEL row: fits and block fences are model-row events
        self.decision_rows = {d.position: d for d in rm.decision_rows}  # decisions are a subset of the model rows outside the fixture
        self.decision_cutoffs = {pos: decision_cutoff(cals, cfg, d.session) for pos, d in self.decision_rows.items()}
        self.decision_positions = tuple(self.decision_rows)
        self.assets = tuple(a.asset_id for a in (cfg.universe.assets or []))
        self.exch = {a.asset_id: a.exchange for a in (cfg.universe.assets or [])}
        self.usd = np.array([a.currency == "USD" for a in (cfg.universe.assets or [])])
        self.rates = {m: expected_cost_rates(cfg, m) for m in (0.0, 1.0, 2.0)}
        self.refits = tuple(range(1, len(rm.eligible) + 1, cfg.evaluation.continuous_refit.every_rows)) \
            if cfg.evaluation.continuous_refit.start_at_first_eligible_decision else ()

    # ------------------------------------------------------------------------------------------ shared, cached work
    def regime_for(self, fit: FoldFit, trial: TrialParams) -> RegimeSource:
        """The fixed table (fixture) is trial-independent; the HMM lane is bound to a fit and to the trial's gate (its state count)."""
        bind = getattr(self.regime, "bind", None)
        return bind(fit, trial) if bind is not None else self.regime

    def decisions_upto(self, last: int) -> int:
        """Number of decisions at model positions 1..last (equals ``last`` on the fixture)."""
        return sum(1 for p in self.decision_positions if p <= last)

    def refit_position(self, position: int) -> int:
        return max(r for r in self.refits if r <= position)

    def fit(self, refit_pos: int, n_components: int | None = None) -> FoldFit:
        key: Any = refit_pos if n_components is None else (refit_pos, n_components)
        if key not in self.caches.fits:
            self.caches.fits[key] = fit_core(self.cfg, self.md, self.cals, self.sess, fold_id="shared", refit_id=f"r{refit_pos}",
                                             fit_end_utc=self.cutoffs[refit_pos - 1], hashes=self.hashes, n_components=n_components)
            self.caches.counts["fits"] += 1
        else:
            self.caches.counts["cache_hits"] += 1
        return self.caches.fits[key]

    @staticmethod
    def _target_subset(t: TrialParams) -> dict[str, Any]:
        """Only target-producing dimensions enter the key; the lambdas shape the optimizer, never the target (spec 3.4)."""
        out = {"overlay_gamma": t.overlay_gamma, "entry_z": t.entry_z, "horizon_weights": list(t.horizon_weights), "gate": t.gate_name}
        if t.n_components is not None:  # research only; the fixture's fingerprint keeps its exact keys
            out["n_components"] = t.n_components
        return out

    def calibration(self, refit_pos: int, trial: TrialParams, fit: FoldFit) -> ObjectiveCalibration:
        fp = build_fingerprint(
            artifact_type="objective_calibration", schema_version="5.3.2", algorithm_component_version="calib-1",
            spec_hash=self.hashes["spec_hash"], code_hash_=self.hashes["code_hash"], lock_hash=None,
            data_manifest_hash=self.hashes["data_manifest_hash"], base_profile_hash=self.hashes["base_profile_hash"],
            asset_order=self.assets, information_cutoff_utc=fmt(self.cutoffs[refit_pos - 1]), fit_start_utc=fmt(fit.fit_start_utc),
            fit_end_utc=fmt(fit.fit_end_utc), observation_set_hash=hash_json_obj([p for p, c in self.decision_cutoffs.items() if c <= fit.fit_end_utc]),
            upstream_content_hashes={"core_fit": fit.key}, relevant_parameter_subset=self._target_subset(trial), seed=None,
            solver_versions={}, numerical_options=default_numerical_options(),
            trial_influence={"influenced": True, "parameters": sorted(self._target_subset(trial)), "trial_hash": None})
        key = cache_key(fp)
        if key not in self.caches.calibrations:
            self.caches.calibrations[key] = calibrate_fold(
                self.cfg, self.md, self.cals, self.sess, fit, self.regime_for(fit, trial), trial,
                [c for c in self.decision_cutoffs.values() if c <= fit.fit_end_utc],
                self._initial_weights(), self.caches.agent_memo)
            self.caches.counts["calibrations"] += 1
        else:
            self.caches.counts["cache_hits"] += 1
        return self.caches.calibrations[key]

    def _initial_weights(self) -> np.ndarray:
        w = np.zeros(len(self.assets) + 1)
        w[-1] = 1.0  # the fixture's profile holdings are 100 % cash (OQ-020)
        return w

    def target(self, position: int, trial: TrialParams, fit: FoldFit) -> DecisionTarget:
        key = (fit.key, position, json_key(self._target_subset(trial)))
        if key not in self.caches.targets:
            cut = self.decision_cutoffs[position]
            ctx = agent_context(self.cfg, self.md, self.cals, self.sess, fit, cut)
            self.caches.targets[key] = (decide(self.cfg, ctx, self.regime_for(fit, trial), trial, cut, self.caches.agent_memo), ctx)
            self.caches.counts["targets"] += 1
        return self.caches.targets[key]

    # ------------------------------------------------------------------------------------------ the path
    def run(self, *, label: str, fold_id: str, trial: TrialParams, trial_hash: str, multiplier: float, start: int, end: int,
            opening: Opening, carry: PathCarry | None = None, execute_until_end_valuation: bool = True,
            policy: BenchmarkPolicy | None = None) -> PathResult:
        """Run decisions ``start..end`` on one isolated ledger. ``carry`` continues a previous committed path exactly."""
        cfg, md, cals = self.cfg, self.md, self.cals
        if carry is None:
            runner = LedgerRunner(cfg, md, cals, opening.cash, dict(opening.holdings), opening.ts, Decimal(str(multiplier)))
            pending: list[Event] = []
            drained = pd.Timestamp.min.tz_localize("UTC")
            last_w = None
        else:
            runner = LedgerRunner.resume(cfg, md, cals, carry.runner, opening.ts, Decimal(str(multiplier)))
            pending, drained, last_w = list(carry.pending), carry.drained_ts, carry.last_weights
            if policy is not None and carry.policy_state is not None:
                policy.state = dict(carry.policy_state)
        heap: list[tuple] = []
        seq = 0
        for e in pending:
            seq += 1
            heapq.heappush(heap, (e.key(), seq, e))
        sidx = int(np.searchsorted([x.ts.value for x in self.static], drained.value, side="right")) if carry else 0

        decisions, fills, plans = [], [], []
        rows: list[dict] = []
        nav: dict[str, Decimal] = {}
        attempts_total = 0
        state = {"sidx": sidx, "seq": seq, "drained": drained}

        def drain(until: pd.Timestamp) -> None:
            while True:
                s = self.static[state["sidx"]] if state["sidx"] < len(self.static) else None
                d = heap[0][2] if heap else None
                pick_static = s is not None and s.ts <= until and (d is None or d.ts > until or s.key() <= d.key())
                pick_dyn = d is not None and d.ts <= until and not pick_static
                if pick_static:
                    state["sidx"] += 1
                    e = s
                elif pick_dyn:
                    e = heapq.heappop(heap)[2]
                else:
                    break
                row = process(e)
                if row:
                    rows.append(row)
                    if e.kind == "valuation":
                        nav[e.order_id.removeprefix("VAL-")] = Decimal(row["nav_krw"])
                state["drained"] = max(state["drained"], e.ts)

        def process(e: Event) -> dict:
            if e.kind not in ("buy_fill", "sell_fill") or "decision" not in e.payload:
                return runner.step(e)
            return fill(e)

        def fill(e: Event) -> dict:
            p = e.payload
            planned = int(p["quantity"])
            origin = p["origin"]
            try:
                md.raw_close_at(e.asset_id, e.ts)
            except NoDataError:
                _, nxt = cals.next_close_after(self.exch[e.asset_id], e.ts)
                if nxt <= utc(p["max_execution_utc"]) and nxt.tz_localize(None) <= self._data_end():
                    state["seq"] += 1
                    moved = Event(nxt, e.kind, e.exchange, e.asset_id, e.order_id, dict(p))
                    heapq.heappush(heap, (moved.key(), state["seq"], moved))
                    fills.append(FillRecord(e.order_id, p["decision"], e.asset_id, fmt(e.ts), planned, 0, "advanced", origin))
                else:
                    fills.append(FillRecord(e.order_id, p["decision"], e.asset_id, fmt(e.ts), planned, 0, "expired", origin))
                return {}
            realized = planned
            status = "filled"
            if planned > 0:
                realized = runner.max_affordable_buy(e, planned)
                status = "filled" if realized == planned else ("reduced_cash" if realized else "rejected_cash")
            else:
                realized = -min(-planned, runner.ledger.holdings.get(e.asset_id, 0))
                status = "filled" if realized == planned else ("reduced_holdings" if realized else "rejected_holdings")
            fills.append(FillRecord(e.order_id, p["decision"], e.asset_id, fmt(e.ts), planned, realized, status, origin))
            return runner.step(e, quantity_override=realized)

        in_block = [p for p in self.decision_positions if start <= p <= end]
        for n_done, pos in enumerate(in_block):
            dec_id = self.decision_rows[pos].decision_id
            cut = self.decision_cutoffs[pos]
            drain(cut)
            runner.accrue_to(cut)
            nav_now = runner.nav_at(cut)
            marks = runner._marks(cut)
            w_minus = np.array([float(Decimal(runner.ledger.holdings[a]) * marks.get(a, Decimal(0)) / nav_now) for a in self.assets]
                               + [float(runner.ledger.cash / nav_now)])
            w_minus[-1] = 1.0 - w_minus[:-1].sum()
            snap = LedgerSnapshot(runner.ledger.cash, dict(runner.ledger.holdings), cut)
            if policy is None:
                rpos = self.refit_position(pos)
                fit = self.fit(rpos, trial.n_components)
                cal = self.calibration(rpos, trial, fit)
                tgt, ctx = self.target(pos, trial, fit)
                inp = OptimizerInputs(tgt.a_bar, w_minus, fit.covariance.matrix, cal.w_track, cal.scales,
                                      (trial.lambda_track, trial.lambda_risk, trial.lambda_cost), self.rates[multiplier],
                                      tgt.combined.b_final, cfg.constraints.target_cash_floor, ctx.max_weights, self.usd,
                                      cfg.constraints.usd_exposure_limit, cfg.constraints.turnover_one_way_limit, ctx.tradable)
                res: OptimizationResult = optimize(inp, cfg, previous_target=last_w, first_decision=(n_done == 0 and carry is None))
                attempts_total += len(res.attempts)
                if res.halted or res.weights is None:
                    raise EngineHalt(f"position {pos}: {res.halt_reason}")
                last_w = res.weights
                plan = build_plan(cfg, md, cals, snap, res.weights, run_id=label, fold_id=fold_id, refit_id=f"r{rpos}",
                                  decision_id=dec_id, trial_hash=trial_hash, decision_cutoff=cut,
                                  multiplier=Decimal(str(multiplier)))
                a_bar, weights_out, b_final = tgt.a_bar.tolist(), res.weights.tolist(), tgt.combined.b_final
                branch, n_attempts, terms = res.branch, len(res.attempts), res.terms
                turnover = float(res.turnover or 0.0)
            else:
                rpos = 0
                ms = mark_state(cfg, md, cals, snap, cut)
                plan, _ = policy.decide(DecisionInput(pos, snap, ms, len(heap), label, fold_id, dec_id, Decimal(str(multiplier)), cals))
                a_bar = [float(x) for x in policy.target()]
                weights_out = [float(Decimal(snap.holdings[a] + next((o.intended_trade_quantity for o in plan.orders if o.asset_id == a), 0))
                                     * ms.unit[a] / ms.nav) for a in self.assets]
                weights_out.append(1.0 - sum(weights_out))
                b_final, branch, n_attempts, terms = float(sum(a_bar[:-1])), f"benchmark:{policy.benchmark_id}", 0, None
                turnover = 0.5 * float(np.abs(np.array(weights_out) - w_minus).sum())
            plans.append(plan)
            n_new = 0
            for o in plan.orders:
                q = o.intended_trade_quantity
                if q == 0:
                    continue
                ts = utc(o.intended_execution_utc)
                ev = Event(ts, "buy_fill" if q > 0 else "sell_fill", self.exch[o.asset_id], o.asset_id, o.order_id,
                           {"quantity": q, "decision": pos, "max_execution_utc": o.max_execution_utc,
                            "origin": {"fold_id": fold_id, "refit_id": f"r{rpos}", "trial_hash": trial_hash,
                                       "decision_id": dec_id}})
                state["seq"] += 1
                heapq.heappush(heap, (ev.key(), state["seq"], ev))
                n_new += 1
            decisions.append(DecisionRecord(
                pos, fmt(cut), rpos, a_bar, weights_out, w_minus.tolist(), b_final, branch, n_attempts, plan.plan_content_hash, n_new,
                plan.summary["shortfall_orders"], turnover, len(heap) - n_new, False,
                None if terms is None else terms.contributions))

        if execute_until_end_valuation:
            drain(self.cutoffs[end - 1])  # the block's information fence is the valuation at its last MODEL row
        carry_out = PathCarry(runner.snapshot(), [h[2] for h in sorted(heap)], state["drained"], last_w,
                              None if policy is None else dict(policy.state))
        return PathResult(label, decisions, fills, nav, pd.DataFrame(rows), carry_out, runner.reconciliation, plans, attempts_total,
                          runner.ledger.cumulative_costs)

    def _data_end(self) -> pd.Timestamp:
        return max(pd.Timestamp(s.event[-1]) for s in self.md.raw.values())


def json_key(obj: Any) -> str:
    return hash_json_obj(obj)


__all__ = [
    "Caches",
    "DecisionRecord",
    "EngineHalt",
    "FillRecord",
    "Opening",
    "PathCarry",
    "PathEngine",
    "PathResult",
    "krw_value",
    "parse_delay",
]
