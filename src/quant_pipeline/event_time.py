"""Global UTC event queue (spec 6.4). One ledger posts one event at a time in this total order."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .calendars import Calendars, RowMap, parse_delay, utc
from .config import Config

PRIORITY = {"corporate_action": 0, "cash_accrual": 1, "sell_fill": 2, "fx_conversion": 3, "buy_fill": 4, "valuation": 5}
EXCHANGE_RANK = {"XKRX": 0, "XNYS": 1, "": 2}


@dataclass
class Event:
    ts: pd.Timestamp
    kind: str
    exchange: str = ""
    asset_id: str = ""
    order_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def key(self) -> tuple:
        return (self.ts, PRIORITY[self.kind], EXCHANGE_RANK[self.exchange], self.asset_id, self.order_id)


def sort_events(events: list[Event]) -> list[Event]:
    """Sort by (timestamp, priority, exchange, asset, order) and assign sequential IDs. Duplicate keys are an error."""
    ordered = sorted(events, key=Event.key)
    keys = [e.key() for e in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate event sort key; order_id must disambiguate")
    for i, e in enumerate(ordered, start=1):
        e.event_id = f"E{i:05d}"
    return ordered


@dataclass(frozen=True)
class FillRequest:
    """A fill to replay at the official close of the asset's exchange on ``session`` (scripted or from a plan)."""

    order_id: str
    asset_id: str
    session: str
    quantity: int


def asset_exchange(cfg: Config) -> dict[str, str]:
    return {a.asset_id: a.exchange for a in (cfg.universe.assets or [])}


def build_events(cfg: Config, cals: Calendars, rm: RowMap, actions: pd.DataFrame, rates: pd.DataFrame,
                 fills: list[FillRequest] | None = None, include_slots: bool = False) -> list[Event]:
    exch = asset_exchange(cfg)
    delay = parse_delay(cfg.availability.price_publication_delay)
    ev: list[Event] = []
    for r in actions.itertuples():
        ex = exch[r.instrument_id]
        ev.append(Event(utc(r.ex_timestamp_utc), "corporate_action", ex, r.instrument_id, f"{r.action_id}-ex",
                        {"action_type": r.action_type, "phase": "ex", "num": r.ratio_num, "den": r.ratio_den,
                         "cash_amount": r.cash_amount}))
        if r.action_type == "cash_dividend":
            ev.append(Event(utc(r.pay_timestamp_utc), "corporate_action", ex, r.instrument_id, f"{r.action_id}-pay",
                            {"action_type": r.action_type, "phase": "pay", "cash_amount": r.cash_amount}))
    for r in rates.itertuples():
        ev.append(Event(utc(r.available_at_utc), "cash_accrual", "", cfg.universe.cash_asset_id, f"RATE-{r.rate_date}",
                        {"annual_rate": r.annual_rate}))
    for s in rm.anchors:
        t = cals.close("XNYS", s) + delay  # valuation uses marks that are causally available
        day = s.strftime("%Y-%m-%d")
        ev.append(Event(t, "cash_accrual", "", cfg.universe.cash_asset_id, f"VAL-ACCR-{day}"))
        ev.append(Event(t, "valuation", "", "PORTFOLIO", f"VAL-{day}"))
    if include_slots:
        for pos, s in enumerate(rm.eligible, start=1):
            cutoff = cals.close("XNYS", s) + delay
            for asset, ex in exch.items():
                _, close = cals.next_close_after(ex, cutoff)
                for side in ("sell", "buy"):
                    ev.append(Event(close, f"{side}_fill", ex, asset, f"D{pos:03d}-{asset}-{side}", {"decision": pos}))
    for f in fills or []:
        ex = exch[f.asset_id]
        close = cals.close(ex, pd.Timestamp(f.session))
        ev.append(Event(close, "buy_fill" if f.quantity > 0 else "sell_fill", ex, f.asset_id, f.order_id,
                        {"quantity": f.quantity}))
    return sort_events(ev)
