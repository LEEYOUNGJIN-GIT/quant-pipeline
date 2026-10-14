"""Causal KRW risk-free cash accrual (spec 4.4). Annual simple quote, ACT/365F, rate known at the interval start."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

from .accounting import quantize_money
from .calendars import Calendars
from .data import NoDataError, StaleDataError


@dataclass(frozen=True)
class RateCurve:
    available: np.ndarray  # datetime64[ns] UTC
    dates: tuple[pd.Timestamp, ...]
    annual: tuple[Decimal, ...]
    max_stale_business_days: int

    @classmethod
    def from_frame(cls, df: pd.DataFrame, max_stale_business_days: int) -> RateCurve:
        df = df.assign(_av=pd.to_datetime(df["available_at_utc"], utc=True)).sort_values("_av")
        if (df["annual_rate"] < 0).any():
            raise ValueError("negative rate; borrowing needs an explicit financing spread (spec 4.4)")
        return cls(df["_av"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                   tuple(pd.Timestamp(d) for d in df["rate_date"]), tuple(Decimal(repr(float(r))) for r in df["annual_rate"]),
                   max_stale_business_days)

    def index_asof(self, ts: pd.Timestamp) -> int:
        return int(np.searchsorted(self.available, np.datetime64(ts.tz_convert("UTC").tz_localize(None)), side="right")) - 1

    def rate_in_force(self, ts: pd.Timestamp, cals: Calendars | None = None) -> Decimal:
        """Latest rate with ``available_at_utc <= ts``; carried at most ``max_stale_business_days`` XKRX sessions."""
        i = self.index_asof(ts)
        if i < 0:
            raise NoDataError(f"no KRW rate available by {ts}")
        if cals is not None:
            day = ts.tz_convert("UTC").tz_localize(None).normalize()
            if day > self.dates[i]:
                carried = len(cals.sessions("XKRX", self.dates[i] + pd.Timedelta(days=1), day))
                if carried > self.max_stale_business_days:
                    raise StaleDataError(f"rate dated {self.dates[i].date()} carried {carried} XKRX business days at {ts}")
        return self.annual[i]


def act_days(t0: pd.Timestamp, t1: pd.Timestamp) -> int:
    """Calendar-date difference. Same-day intervals accrue nothing (provisional convention, see OQ-008)."""
    d0 = t0.tz_convert("UTC").tz_localize(None).normalize()
    d1 = t1.tz_convert("UTC").tz_localize(None).normalize()
    if d1 < d0:
        raise ValueError("accrual interval runs backwards")
    return (d1 - d0).days


def accrue(cash: Decimal, annual_rate: Decimal, t0: pd.Timestamp, t1: pd.Timestamp, places: int = 2) -> Decimal:
    """cash x (1 + r x ACT_days / 365). Only positive cash accrues; negative cash is prohibited upstream."""
    if cash < 0:
        raise ValueError("negative cash cannot accrue at the deposit rate")
    days = act_days(t0, t1)
    if cash == 0 or annual_rate == 0 or days == 0:
        return cash
    return quantize_money(cash * (1 + annual_rate * Decimal(days) / Decimal(365)), places)
