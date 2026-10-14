"""The four labelled benchmarks (spec 10.3, 10.3.1) and the pro-rata partial-trade rule.

Every rules benchmark uses the strategy's decision cutoffs, next-close rule, calendars, tradability, lots, marks, FX, cost model
and ledger. Only the conventional buy-and-hold reference is exempt from the ordinary turnover limit, and only at its first decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any

import numpy as np

from .calendars import Calendars
from .config import Config
from .rebalance import (
    LedgerSnapshot,
    MarkState,
    RebalancePlan,
    assemble_plan,
    projected_cash_after,
)

LABELS = {
    "cash_krw": "KRW cash reference",
    "equal_weight_periodic": "turnover-matched periodic equal weight",
    "equal_weight_staged_entry": "turnover-matched staged-entry equal-weight hold",
    "equal_weight_buy_and_hold_reference": "conventional buy-and-hold reference — initial turnover exempt",
}
COMPARABILITY = {"cash_krw": "financing_context", "equal_weight_periodic": "implementation_matched",
                 "equal_weight_staged_entry": "implementation_matched",
                 "equal_weight_buy_and_hold_reference": "reference_initial_turnover_exempt"}


class BenchmarkInfeasible(ValueError):
    """A rules benchmark cannot be built from the configured universe: halt before evaluation, never drop members silently."""


def validate_benchmarks(cfg: Config) -> None:
    members = [a for a in (cfg.universe.assets or []) if a.benchmark_eligible]
    for b in cfg.benchmarks.definitions:
        if b.type != "rules":
            continue
        w = b.total_risky_weight or 0.0
        if len(members) < cfg.benchmarks.required_benchmark_eligible_count:
            raise BenchmarkInfeasible(f"{b.benchmark_id}: |M|={len(members)} below the required count")
        if sum(a.max_weight for a in members) < w:
            raise BenchmarkInfeasible(f"{b.benchmark_id}: member caps cannot cover B={w}")
        if w / len(members) > min(a.max_weight for a in members):
            raise BenchmarkInfeasible(f"{b.benchmark_id}: equal weight {w / len(members):.4f} exceeds the smallest member cap")


def equal_weight_target(cfg: Config, total_risky: float) -> np.ndarray:
    assets = list(cfg.universe.assets or [])
    members = [a.benchmark_eligible for a in assets]
    w = np.array([total_risky / sum(members) if m else 0.0 for m in members])
    return np.append(w, 1.0 - w.sum())


def _weights(ms: MarkState, holdings: dict[str, int]) -> np.ndarray:
    risky = [float(Decimal(holdings.get(a.asset_id, 0)) * ms.unit[a.asset_id] / ms.nav) for a in ms.assets]
    return np.array([*risky, 0.0])


@dataclass(frozen=True)
class ProRataInfo:
    required_turnover: float
    scale: float
    exempt: bool
    buy_scale: float
    residual_lots_added: tuple[str, ...]
    residual_lots_removed: tuple[str, ...]
    no_trade_reason: str | None


def prorata_trade(cfg: Config, ms: MarkState, snapshot: LedgerSnapshot, target: np.ndarray, *, turnover_limit: float | None,
                  multiplier: Decimal) -> tuple[dict[str, int], dict[str, int], dict[str, str], ProRataInfo]:
    """Spec 10.3.1, steps 1-8. ``turnover_limit=None`` means the reference's one-time initial exemption."""
    assets = ms.assets
    n = len(assets)
    w_before = _weights(ms, snapshot.holdings)
    w_before[-1] = float(ms.cash / ms.nav)
    required = 0.5 * float(np.abs(target - w_before).sum())
    exempt = turnover_limit is None
    scale = 1.0 if exempt or required <= 0 else min(1.0, turnover_limit / required)               # step 1
    floor = Decimal(str(cfg.constraints.minimum_post_trade_cash_base))
    unconstrained, desired_value = {}, {}
    for i, a in enumerate(assets):                                                                   # steps 2-3
        pre = snapshot.holdings.get(a.asset_id, 0)
        gap_w = target[i] - w_before[i]
        value = Decimal(repr(float(gap_w * scale))) * ms.nav if ms.tradable[a.asset_id] else Decimal(0)
        desired_value[a.asset_id] = value
        lots = (abs(value) / (ms.unit[a.asset_id] * a.lot_size)).to_integral_value(rounding=ROUND_DOWN)
        q = int(lots) * a.lot_size * (1 if value >= 0 else -1)
        unconstrained[a.asset_id] = max(q, -pre)
    trade = dict(unconstrained)
    reasons: dict[str, str] = {}
    if scale < 1.0:
        reasons.update({a.asset_id: "turnover_scaling" for a in assets if trade[a.asset_id]})

    buy_scale = 1.0                                                                                  # step 4
    if projected_cash_after(cfg, ms, trade, multiplier) < floor:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            probe = {a.asset_id: (int(Decimal(repr(mid * trade[a.asset_id])) / a.lot_size) * a.lot_size if trade[a.asset_id] > 0
                                  else trade[a.asset_id]) for a in assets}
            lo, hi = (mid, hi) if projected_cash_after(cfg, ms, probe, multiplier) >= floor else (lo, mid)
        buy_scale = lo
        for a in assets:
            if trade[a.asset_id] > 0:
                trade[a.asset_id] = int(Decimal(repr(buy_scale * trade[a.asset_id])) / a.lot_size) * a.lot_size
                reasons[a.asset_id] = "cash_pro_rata"

    def feasible(tr: dict[str, int]) -> bool:                                                        # step 6
        if projected_cash_after(cfg, ms, tr, multiplier) < floor:
            return False
        after = np.array([float(Decimal(snapshot.holdings.get(a.asset_id, 0) + tr[a.asset_id]) * ms.unit[a.asset_id] / ms.nav)
                          for a in assets])
        if any(after[i] > assets[i].max_weight + 1e-12 for i in range(n)):
            return False
        if sum(after[i] for i in range(n) if assets[i].currency == "USD") > cfg.constraints.usd_exposure_limit + 1e-12:
            return False
        if any(tr[a.asset_id] and not ms.tradable[a.asset_id] for a in assets):
            return False
        cash_w = float(projected_cash_after(cfg, ms, tr, multiplier) / ms.nav)
        turn = 0.5 * (np.abs(after - w_before[:n]).sum() + abs(cash_w - w_before[-1]))
        return exempt or turn <= turnover_limit + 1e-12

    added: list[str] = []                                                                            # step 5
    changed = True
    while changed:
        changed = False
        gaps = sorted(((desired_value[a.asset_id] - Decimal(trade[a.asset_id]) * ms.unit[a.asset_id], a) for a in assets
                       if ms.tradable[a.asset_id] and desired_value[a.asset_id] > 0),
                      key=lambda g: (-g[0], assets.index(g[1])))
        for gap, a in gaps:
            if gap < ms.unit[a.asset_id] * a.lot_size * Decimal("0.0001"):
                continue
            probe = {**trade, a.asset_id: trade[a.asset_id] + a.lot_size}
            if feasible(probe):
                trade = probe
                added.append(a.asset_id)
                changed = True
    removed: list[str] = []
    while not feasible(trade) and added:                                                             # step 7
        a_id = added.pop()
        trade[a_id] -= next(x.lot_size for x in assets if x.asset_id == a_id)
        removed.append(a_id)
    reason = None if any(trade.values()) else ("already_at_target" if required <= 1e-9 else "no_feasible_nonzero_order")  # step 8
    return trade, unconstrained, reasons, ProRataInfo(required, scale, exempt, buy_scale, tuple(added), tuple(removed), reason)


@dataclass
class DecisionInput:
    position: int
    snapshot: LedgerSnapshot
    mark_state: MarkState
    pending_orders: int
    run_id: str
    fold_id: str
    decision_id: str
    multiplier: Decimal
    cals: Calendars


@dataclass
class BenchmarkPolicy:
    """Base: ``decide`` returns an immutable plan (possibly empty) plus policy diagnostics. State is part of the carry."""

    cfg: Config
    benchmark_id: str
    state: dict[str, Any] = field(default_factory=dict)

    def target(self) -> np.ndarray:
        b = next(d for d in self.cfg.benchmarks.definitions if d.benchmark_id == self.benchmark_id)
        return equal_weight_target(self.cfg, b.total_risky_weight or 0.0)

    def _plan(self, di: DecisionInput, trade, unconstrained, reasons, tgt, extra) -> RebalancePlan:
        return assemble_plan(self.cfg, di.cals, di.mark_state, di.snapshot, tgt, trade, unconstrained, reasons, run_id=di.run_id,
                             fold_id=di.fold_id, refit_id="none", decision_id=di.decision_id, trial_hash=f"benchmark:{self.benchmark_id}",
                             multiplier=di.multiplier, extra_summary=extra)

    def decide(self, di: DecisionInput) -> tuple[RebalancePlan, dict[str, Any]]:  # pragma: no cover - overridden
        raise NotImplementedError


class CashPolicy(BenchmarkPolicy):
    def decide(self, di: DecisionInput):
        zero = {a.asset_id: 0 for a in di.mark_state.assets}
        tgt = np.append(np.zeros(len(zero)), 1.0)
        return self._plan(di, zero, zero, {}, tgt, {"benchmark": self.benchmark_id}), {"no_trade_reason": "cash_reference"}


class PeriodicEqualWeight(BenchmarkPolicy):
    def decide(self, di: DecisionInput):
        tgt = self.target()
        trade, unc, reasons, info = prorata_trade(self.cfg, di.mark_state, di.snapshot, tgt,
                                                  turnover_limit=self.cfg.constraints.turnover_one_way_limit, multiplier=di.multiplier)
        return self._plan(di, trade, unc, reasons, tgt, {"benchmark": self.benchmark_id, "prorata": info.__dict__}), info.__dict__


class StagedEntry(BenchmarkPolicy):
    """Moves toward the equal-weight target under the ordinary turnover limit; after the first attainment it never rebalances."""

    def decide(self, di: DecisionInput):
        tgt = self.target()
        st = self.state
        st.setdefault("attained", False)
        st.setdefault("active_decisions", 0)
        st.setdefault("stalled", False)
        ms = di.mark_state
        tol = self.cfg.benchmarks.target_attainment_tolerance
        w = _weights(ms, di.snapshot.holdings)
        w[-1] = float(ms.cash / ms.nav)
        errors = {"turnover_to_target": 0.5 * float(np.abs(tgt - w).sum()), "max_asset_error": float(np.abs(tgt[:-1] - w[:-1]).max()),
                  "risky_gross_error": abs(float(tgt[:-1].sum() - w[:-1].sum()))}
        if not st["attained"] and di.pending_orders == 0 and max(errors.values()) <= tol:
            st["attained"] = True
            st["attained_at_position"] = di.position
        zero = {a.asset_id: 0 for a in ms.assets}
        if st["attained"]:
            return self._plan(di, zero, zero, {}, tgt, {"benchmark": self.benchmark_id, "attained": True, "errors": errors}), \
                {"no_trade_reason": "target_attained_hold", "errors": errors}
        st["active_decisions"] += 1
        if st["active_decisions"] > self.cfg.benchmarks.staged_entry_max_active_decisions:
            st["stalled"] = True  # the rule is unchanged: no exemption is granted
        trade, unc, reasons, info = prorata_trade(self.cfg, ms, di.snapshot, tgt,
                                                  turnover_limit=self.cfg.constraints.turnover_one_way_limit, multiplier=di.multiplier)
        extra = {"benchmark": self.benchmark_id, "attained": False, "errors": errors, "active_decisions": st["active_decisions"],
                 "staged_entry_stalled": st["stalled"], "prorata": info.__dict__}
        return self._plan(di, trade, unc, reasons, tgt, extra), {**info.__dict__, "staged_entry_stalled": st["stalled"]}


class BuyAndHoldReference(BenchmarkPolicy):
    """One initial shared decision; only that decision is exempt from the ordinary turnover cap (net deployment only)."""

    def decide(self, di: DecisionInput):
        tgt = self.target()
        zero = {a.asset_id: 0 for a in di.mark_state.assets}
        if self.state.get("initial_done"):
            return self._plan(di, zero, zero, {}, tgt, {"benchmark": self.benchmark_id, "initial_done": True}), {"no_trade_reason": "hold"}
        trade, unc, reasons, info = prorata_trade(self.cfg, di.mark_state, di.snapshot, tgt, turnover_limit=None, multiplier=di.multiplier)
        self.state["initial_done"] = True
        self.state["initial_turnover"] = info.required_turnover
        extra = {"benchmark": self.benchmark_id, "comparability": COMPARABILITY[self.benchmark_id], "initial_turnover_exemption": True,
                 "prorata": info.__dict__}
        return self._plan(di, trade, unc, reasons, tgt, extra), info.__dict__


def make_policy(cfg: Config, benchmark_id: str) -> BenchmarkPolicy:
    cls = {"cash_krw": CashPolicy, "equal_weight_periodic": PeriodicEqualWeight, "equal_weight_staged_entry": StagedEntry,
           "equal_weight_buy_and_hold_reference": BuyAndHoldReference}[benchmark_id]
    return cls(cfg, benchmark_id)
