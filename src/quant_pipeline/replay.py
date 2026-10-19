"""Sequential KRX/NYSE ledger (spec 6.4, 10.1). Each event reads only the state left by earlier events.

``LedgerRunner`` is the stateful engine: it can be stepped event by event (so a decision can read the state after all earlier
events and before any later one) and snapshotted for carry across fold boundaries. ``replay`` is the batch wrapper.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from .accounting import CostRates, Fill, Ledger, quantize_money
from .calendars import Calendars, fmt
from .config import Config
from .data import MarketData
from .event_time import Event, asset_exchange
from .fx import krw_value
from .rates import RateCurve, accrue
from .reconciliation import split_quantity

ZERO = Decimal(0)


@dataclass
class ReplayResult:
    rows: pd.DataFrame
    reconciliation: list[dict]
    ledger: Ledger
    pending_dividends: dict[str, Decimal] = field(default_factory=dict)


def cost_rates(cfg: Config, exchange: str, multiplier: Decimal = Decimal(1)) -> CostRates:
    c = cfg.costs
    return CostRates(Decimal(str(c.commission_bps)), Decimal(str(c.half_spread_slippage_bps)),
                     Decimal(str(c.sell_tax_bps[exchange])), Decimal(str(c.fx_conversion_bps)),
                     Decimal(str(c.minimum_fee_base))).scaled(multiplier)


@dataclass
class RunnerState:
    """Everything that must survive a fold boundary: quantities, cash, accrual clock and dividend entitlements."""

    cash: Decimal
    holdings: dict[str, int]
    last_ts: pd.Timestamp
    cumulative_costs: Decimal
    entitlement: dict[str, int]


class LedgerRunner:
    def __init__(self, cfg: Config, md: MarketData, cals: Calendars, opening_cash: Decimal, opening_qty: dict[str, int],
                 opening_ts: pd.Timestamp, multiplier: Decimal = Decimal(1)):
        self.cfg, self.md, self.cals, self.multiplier = cfg, md, cals, multiplier
        self.exch = asset_exchange(cfg)
        self.ccy = {a.asset_id: a.currency for a in (cfg.universe.assets or [])}
        self.places = cfg.accounting.money_decimal_places
        self.curve = RateCurve.from_frame(md.rates, cfg.availability.max_rate_stale_business_days)
        self.ledger = Ledger(cash=opening_cash, holdings={a: opening_qty.get(a, 0) for a in self.exch}, money_places=self.places,
                             allow_negative_cash=cfg.cash.allow_negative)
        self.opening_ts = opening_ts
        self.last_ts = opening_ts
        self.entitlement: dict[str, int] = {}
        self.reconciliation: list[dict] = []

    # ------------------------------------------------------------------------------------------ state carry
    def snapshot(self) -> RunnerState:
        return RunnerState(self.ledger.cash, dict(self.ledger.holdings), self.last_ts, self.ledger.cumulative_costs,
                           dict(self.entitlement))

    @classmethod
    def resume(cls, cfg: Config, md: MarketData, cals: Calendars, state: RunnerState, opening_ts: pd.Timestamp,
               multiplier: Decimal = Decimal(1)) -> LedgerRunner:
        r = cls(cfg, md, cals, state.cash, dict(state.holdings), opening_ts, multiplier)
        r.last_ts, r.ledger.cumulative_costs, r.entitlement = state.last_ts, state.cumulative_costs, dict(state.entitlement)
        return r

    # ------------------------------------------------------------------------------------------ marks
    def _marks(self, ts: pd.Timestamp, override: dict[str, Decimal] | None = None) -> dict[str, Decimal]:
        fx = None
        out: dict[str, Decimal] = {}
        for a, q in self.ledger.holdings.items():
            if not q:
                continue
            if override and a in override:
                out[a] = override[a]
                continue
            price = self.md.raw_mark(a, ts)
            if self.ccy[a] == "USD":
                fx = fx or self.md.fx_mark(ts)
            out[a] = krw_value(Decimal(1), price, self.ccy[a], fx if self.ccy[a] == "USD" else None)
        return out

    @staticmethod
    def _nav(cash: Decimal, qty: dict[str, int], marks: dict[str, Decimal]) -> Decimal:
        return cash + sum((Decimal(q) * marks[a] for a, q in qty.items() if q), ZERO)

    def accrue_to(self, ts: pd.Timestamp) -> None:
        """Accrue positive cash from the previous event at the rate in force then (spec 4.4)."""
        if ts > self.last_ts and self.ledger.cash > 0:
            self.ledger.cash = accrue(self.ledger.cash, self.curve.rate_in_force(self.last_ts, self.cals), self.last_ts, ts, self.places)
        self.last_ts = max(self.last_ts, ts)

    def nav_at(self, ts: pd.Timestamp) -> Decimal:
        return quantize_money(self._nav(self.ledger.cash, self.ledger.holdings, self._marks(ts)), self.places)

    # ------------------------------------------------------------------------------------------ one event
    def step(self, e: Event, quantity_override: int | None = None) -> dict:
        """Apply one event and return its ledger row. ``quantity_override`` is the realized (feasibility-reduced) fill size."""
        if e.ts < self.opening_ts:
            return {}
        self.accrue_to(e.ts)
        note, nav = "", ""
        if e.kind == "corporate_action":
            p = e.payload
            if p["phase"] == "ex":
                self.entitlement[e.order_id.removesuffix("-ex")] = self.ledger.holdings.get(e.asset_id, 0)
                if p["action_type"] == "split":
                    self.ledger.holdings[e.asset_id] = split_quantity(
                        self.ledger.holdings[e.asset_id], int(p["num"]), int(p["den"]),
                        self.cfg.accounting.corporate_action_reconciliation.split_relative_tolerance)  # level 1: halts on a residual above tolerance
                    note = "split_quantity_adjust"
            else:  # dividend pay: the quantity held at the ex event; a USD dividend is credited in KRW at the latest available FX, a KRW one as is
                fx = self.md.fx_mark(e.ts) if self.ccy[e.asset_id] == "USD" else Decimal(1)
                held = self.entitlement.get(e.order_id.removesuffix("-pay"), 0)
                self.ledger.cash = quantize_money(self.ledger.cash + Decimal(held) * Decimal(str(p["cash_amount"])) * fx, self.places)
                note = f"dividend_credit_fx={fx}"
        elif e.kind in ("buy_fill", "sell_fill") and "quantity" in e.payload:
            qty = int(e.payload["quantity"]) if quantity_override is None else quantity_override
            if qty != 0:
                price = self.md.raw_close_at(e.asset_id, e.ts)
                fx = self.md.fx_mark(e.ts) if self.ccy[e.asset_id] == "USD" else None
                rates = cost_rates(self.cfg, self.exch[e.asset_id], self.multiplier)
                before_cash, before_qty = self.ledger.cash, dict(self.ledger.holdings)
                ref = krw_value(Decimal(1), price, self.ccy[e.asset_id], fx)
                mk = self._marks(e.ts, {e.asset_id: ref})
                mk.setdefault(e.asset_id, ref)
                nav_before = self._nav(before_cash, before_qty, mk)
                res = self.ledger.post_fill(Fill(e.asset_id, self.ccy[e.asset_id], qty, price, fx, rates.sell_tax_bps), rates)
                residual = self._nav(self.ledger.cash, self.ledger.holdings, mk) - nav_before + res["cost"]
                tol = max(Decimal("0.01"), Decimal("1e-12") * abs(nav_before))
                self.reconciliation.append({"order_id": e.order_id, "cost": res["cost"], "nav_change_plus_cost": residual,
                                            "within_tolerance": abs(residual) <= tol})
                note = f"px={price};fx={fx};gross={res['notional']};cost={res['cost']}"
        elif e.kind == "valuation":
            nav = str(self.nav_at(e.ts))
        return {"event_id": e.event_id, "ts": fmt(e.ts), "kind": e.kind, "order_id": e.order_id, "cash_krw": str(self.ledger.cash),
                **{f"qty_{a}": self.ledger.holdings[a] for a in self.exch}, "cumulative_costs": str(self.ledger.cumulative_costs),
                "nav_krw": nav, "note": note}

    def max_affordable_buy(self, e: Event, wanted: int) -> int:
        """Largest buy (in lots) whose realized cost still leaves cash >= the post-trade floor (spec 6.4)."""
        lot = next(a.lot_size for a in (self.cfg.universe.assets or []) if a.asset_id == e.asset_id)
        floor = Decimal(str(self.cfg.constraints.minimum_post_trade_cash_base))
        self.accrue_to(e.ts)
        price = self.md.raw_close_at(e.asset_id, e.ts)
        fx = self.md.fx_mark(e.ts) if self.ccy[e.asset_id] == "USD" else None
        rates = cost_rates(self.cfg, self.exch[e.asset_id], self.multiplier)
        from .accounting import trade_cost  # local: avoids widening the module's import surface

        def ok(q: int) -> bool:
            gross = quantize_money(krw_value(Decimal(q), price, self.ccy[e.asset_id], fx), self.places)
            return self.ledger.cash - gross - trade_cost(gross, rates, self.ccy[e.asset_id] == "USD", False, self.places) >= floor

        lo, hi = 0, wanted // lot
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid * lot):
                lo = mid
            else:
                hi = mid - 1
        return lo * lot

    def deep_copy_state(self) -> RunnerState:
        return copy.deepcopy(self.snapshot())


def replay(cfg: Config, md: MarketData, cals: Calendars, events: list[Event], opening_cash: Decimal,
           opening_qty: dict[str, int], opening_ts: pd.Timestamp, multiplier: Decimal = Decimal(1)) -> ReplayResult:
    runner = LedgerRunner(cfg, md, cals, opening_cash, opening_qty, opening_ts, multiplier)
    rows = [r for r in (runner.step(e) for e in events) if r]
    return ReplayResult(pd.DataFrame(rows), runner.reconciliation, runner.ledger)
