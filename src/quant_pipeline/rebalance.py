"""Immutable rebalance plan (spec 6.3, 6.4, 9.5): raw-price sizing at the decision mark, lot rounding toward zero,
projected costs/cash and an explicit execution shortfall. The plan is never rewritten after prices are observed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .accounting import quantize_money, trade_cost
from .calendars import Calendars, fmt, parse_delay, utc
from .config import Config
from .data import MarketData
from .fx import krw_value
from .hashing import hash_json_obj
from .rates import RateCurve, accrue

ZERO = Decimal(0)


@dataclass(frozen=True)
class LedgerSnapshot:
    cash: Decimal
    holdings: dict[str, int]
    as_of: pd.Timestamp


@dataclass(frozen=True)
class PlanOrder:
    order_id: str
    asset_id: str
    side: str
    target_absolute_weight: str
    pretrade_quantity: int
    intended_quantity: int
    intended_trade_quantity: int
    unconstrained_trade_quantity: int
    shortfall_quantity: int
    shortfall_reason: str
    lot_size: int
    sizing_price_raw: str
    sizing_fx: str
    expected_trade_value_krw: str
    expected_cost_krw: str
    intended_execution_utc: str
    max_execution_utc: str
    status: str


@dataclass(frozen=True)
class RebalancePlan:
    run_id: str
    fold_id: str
    refit_id: str
    decision_id: str
    trial_hash: str
    decision_cutoff_utc: str
    nav_pre_cost_krw: str
    cash_accrued_krw: str
    projected_post_cost_cash_krw: str
    target_weights: dict[str, str]
    orders: tuple[PlanOrder, ...]
    no_trade_reason: str | None
    summary: dict[str, Any]
    plan_content_hash: str

    def frame(self) -> pd.DataFrame:
        base = {"run_id": self.run_id, "fold_id": self.fold_id, "refit_id": self.refit_id, "decision_id": self.decision_id,
                "trial_hash": self.trial_hash, "decision_cutoff_utc": self.decision_cutoff_utc,
                "plan_content_hash": self.plan_content_hash}
        return pd.DataFrame([{**base, **o.__dict__} for o in self.orders])


def _content(plan_fields: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(plan_fields, default=str))


def _trunc_lots(value_krw: Decimal, unit_krw: Decimal, lot: int) -> int:
    """Quantity (a lot multiple) whose KRW value does not exceed ``value_krw``; rounded toward zero."""
    if unit_krw <= 0:
        raise ValueError("non-positive unit value")
    lots = (value_krw / (unit_krw * lot)).to_integral_value(rounding=ROUND_DOWN)
    return int(lots) * lot


@dataclass(frozen=True)
class MarkState:
    """Decision-time marks: accrued cash, per-asset raw price / FX / KRW unit value, tradability and pre-cost NAV."""

    assets: tuple
    cutoff: pd.Timestamp
    cash: Decimal
    nav: Decimal
    price: dict[str, Decimal]
    fx: dict[str, Decimal | None]
    unit: dict[str, Decimal]
    tradable: dict[str, bool]
    places: int


def mark_state(cfg: Config, md: MarketData, cals: Calendars, snapshot: LedgerSnapshot, decision_cutoff: pd.Timestamp) -> MarkState:
    assets = tuple(cfg.universe.assets or [])
    cutoff = utc(decision_cutoff)
    places = cfg.accounting.money_decimal_places
    curve = RateCurve.from_frame(md.rates, cfg.availability.max_rate_stale_business_days)
    cash = snapshot.cash
    if cutoff > snapshot.as_of:
        cash = accrue(cash, curve.rate_in_force(snapshot.as_of, cals), snapshot.as_of, cutoff, places)
    price, fx, unit, tradable = {}, {}, {}, {}
    nav = cash
    for a in assets:
        price[a.asset_id] = md.raw_mark(a.asset_id, cutoff)
        fx[a.asset_id] = md.fx_mark(cutoff) if a.currency == "USD" else None
        unit[a.asset_id] = krw_value(Decimal(1), price[a.asset_id], a.currency, fx[a.asset_id])
        tradable[a.asset_id] = md.tradable(a.asset_id, cutoff)
        nav += Decimal(snapshot.holdings.get(a.asset_id, 0)) * unit[a.asset_id]
    return MarkState(assets, cutoff, cash, quantize_money(nav, places), price, fx, unit, tradable, places)


def cost_of(cfg: Config, ms: MarkState, asset, quantity: int, multiplier: Decimal) -> Decimal:
    from .replay import cost_rates

    if quantity == 0:
        return ZERO
    gross = quantize_money(abs(Decimal(quantity)) * ms.unit[asset.asset_id], ms.places)
    return trade_cost(gross, cost_rates(cfg, asset.exchange, multiplier), asset.currency == "USD", quantity < 0, ms.places)


def projected_cash_after(cfg: Config, ms: MarkState, trade: dict[str, int], multiplier: Decimal) -> Decimal:
    c = ms.cash
    for a in ms.assets:
        q = trade[a.asset_id]
        if q:
            gross = quantize_money(abs(Decimal(q)) * ms.unit[a.asset_id], ms.places)
            cost = cost_of(cfg, ms, a, q, multiplier)
            c += (gross - cost) if q < 0 else -(gross + cost)
    return c


def assemble_plan(cfg: Config, cals: Calendars, ms: MarkState, snapshot: LedgerSnapshot, target: np.ndarray, trade: dict[str, int],
                  unconstrained: dict[str, int], reasons: dict[str, str], *, run_id: str, fold_id: str, refit_id: str,
                  decision_id: str, trial_hash: str, multiplier: Decimal, extra_summary: dict[str, Any] | None = None) -> RebalancePlan:
    """Turn final integer trades into the immutable plan (spec 6.3). Every reduction stays visible as shortfall."""
    assets, cutoff, places = ms.assets, ms.cutoff, ms.places
    latency = parse_delay(cfg.decision_schedule.order_latency)
    orders = []
    for i, a in enumerate(assets):
        q, q0 = trade[a.asset_id], unconstrained[a.asset_id]
        if q == 0 and q0 == 0:
            continue
        sess, close = cals.next_close_after(a.exchange, cutoff + latency)
        last = cals.cals[a.exchange].session_offset(sess, cfg.decision_schedule.max_order_age_sessions)
        gross = quantize_money(abs(Decimal(q)) * ms.unit[a.asset_id], places)
        short = abs(q0) - abs(q)
        orders.append(PlanOrder(
            order_id=f"{run_id}:{decision_id}:{a.asset_id}", asset_id=a.asset_id, side="buy" if q0 > 0 else "sell",
            target_absolute_weight=repr(float(target[i])), pretrade_quantity=snapshot.holdings.get(a.asset_id, 0),
            intended_quantity=snapshot.holdings.get(a.asset_id, 0) + q, intended_trade_quantity=q, unconstrained_trade_quantity=q0,
            shortfall_quantity=short, shortfall_reason=reasons.get(a.asset_id, "") if short else "", lot_size=a.lot_size,
            sizing_price_raw=str(ms.price[a.asset_id]), sizing_fx="" if ms.fx[a.asset_id] is None else str(ms.fx[a.asset_id]),
            expected_trade_value_krw=str(gross if q >= 0 else -gross), expected_cost_krw=str(cost_of(cfg, ms, a, q, multiplier)),
            intended_execution_utc=fmt(close), max_execution_utc=fmt(cals.close(a.exchange, last)),
            status="rejected_cash_feasibility" if q == 0 else "planned"))
    weights = {a.asset_id: repr(float(target[i])) for i, a in enumerate(assets)}
    weights[cfg.universe.cash_asset_id] = repr(float(target[-1]))
    held_value = sum((Decimal(snapshot.holdings.get(a.asset_id, 0) + trade[a.asset_id]) * ms.unit[a.asset_id] for a in assets), ZERO)
    summary = {"target_risky_gross": float(sum(target[:-1])), "intended_risky_gross": float(held_value / ms.nav) if ms.nav else 0.0,
               "shortfall_orders": [o.asset_id for o in orders if o.shortfall_quantity],
               "untradable": [k for k, v in ms.tradable.items() if not v], "cost_multiplier": str(multiplier), "money_places": places,
               **(extra_summary or {})}
    projected = projected_cash_after(cfg, ms, trade, multiplier)
    body = _content({"run_id": run_id, "fold_id": fold_id, "refit_id": refit_id, "decision_id": decision_id, "trial_hash": trial_hash,
                     "decision_cutoff_utc": fmt(cutoff), "nav": str(ms.nav), "cash": str(ms.cash), "projected_cash": str(projected),
                     "weights": weights, "orders": [o.__dict__ for o in orders], "summary": summary})
    return RebalancePlan(run_id, fold_id, refit_id, decision_id, trial_hash, fmt(cutoff), str(ms.nav), str(ms.cash), str(projected),
                         weights, tuple(orders), None if orders else "no_trade_required", summary, hash_json_obj(body))


def build_plan(cfg: Config, md: MarketData, cals: Calendars, snapshot: LedgerSnapshot, target: np.ndarray, *, run_id: str,
               fold_id: str, refit_id: str, decision_id: str, trial_hash: str, decision_cutoff: pd.Timestamp,
               multiplier: Decimal = Decimal(1)) -> RebalancePlan:
    """Convert continuous target weights into integer lot-sized orders and persist every reduction as shortfall."""
    ms = mark_state(cfg, md, cals, snapshot, decision_cutoff)
    assets = ms.assets
    wanted: dict[str, int] = {}
    for i, a in enumerate(assets):
        pre = snapshot.holdings.get(a.asset_id, 0)
        if not ms.tradable[a.asset_id]:
            wanted[a.asset_id] = pre  # an untradable asset keeps its quantity; no order is created
            continue
        wanted[a.asset_id] = max(_trunc_lots(Decimal(repr(float(target[i]))) * ms.nav, ms.unit[a.asset_id], a.lot_size), 0)
    trade = {a.asset_id: wanted[a.asset_id] - snapshot.holdings.get(a.asset_id, 0) for a in assets}
    unconstrained = dict(trade)
    reasons: dict[str, str] = {}
    floor = Decimal(str(cfg.constraints.minimum_post_trade_cash_base))
    # deterministic cash-feasibility reduction: lowest-priority buy first (spec 6.4); feasibility is monotone in the
    # reduction, so a binary search per asset equals decrementing one lot at a time
    buys = sorted((a for a in assets if trade[a.asset_id] > 0),
                  key=lambda a: (-Decimal(trade[a.asset_id]) * ms.unit[a.asset_id], assets.index(a)))
    for a in reversed(buys):
        if projected_cash_after(cfg, ms, trade, multiplier) >= floor:
            break
        lot, hi = a.lot_size, trade[a.asset_id] // a.lot_size
        lo = 0
        while lo < hi:  # largest number of lots that still leaves projected cash >= floor
            mid = (lo + hi + 1) // 2
            if projected_cash_after(cfg, ms, {**trade, a.asset_id: mid * lot}, multiplier) >= floor:
                lo = mid
            else:
                hi = mid - 1
        reasons[a.asset_id] = "cash_feasibility"
        trade[a.asset_id] = lo * lot
    if projected_cash_after(cfg, ms, trade, multiplier) < floor:
        raise RuntimeError("no self-financing path: even sell-only orders breach the post-trade cash floor")
    return assemble_plan(cfg, cals, ms, snapshot, target, trade, unconstrained, reasons, run_id=run_id, fold_id=fold_id,
                         refit_id=refit_id, decision_id=decision_id, trial_hash=trial_hash, multiplier=multiplier)


def write_plan(plan: RebalancePlan, directory: Path) -> Path:
    """Write ``rebalance_plan.parquet`` and its canonical JSON manifest once; an existing plan is never overwritten."""
    directory = Path(directory)
    manifest = directory / "rebalance_plan.manifest.json"
    if manifest.exists() or (directory / "rebalance_plan.parquet").exists():
        raise FileExistsError(f"rebalance plan already exists and is immutable: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    frame = plan.frame()
    if frame.empty:
        frame = pd.DataFrame(columns=["run_id", "plan_content_hash"])
    frame.to_parquet(directory / "rebalance_plan.parquet", index=False)
    body = {k: v for k, v in plan.__dict__.items() if k != "orders"}
    body["order_ids"] = [o.order_id for o in plan.orders]
    manifest.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    return manifest
