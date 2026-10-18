"""Fixture/snapshot input loading and causal as-of access (spec 4.2, 4.4). Causality is by ``available_at_utc``."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


class StaleDataError(RuntimeError):
    """A causally available quote exists but is older than the configured maximum (halt, spec 4.4/4.5)."""


class NoDataError(RuntimeError):
    """No observation with ``available_at_utc <= cutoff`` exists."""


@dataclass(frozen=True)
class Series:
    """One instrument's observations sorted by event time with a parallel availability array."""

    event: np.ndarray  # datetime64[ns], UTC
    available: np.ndarray
    value: np.ndarray
    quality: np.ndarray

    def asof_index(self, cutoff: pd.Timestamp) -> int:
        """Index of the latest observation with ``available_at_utc <= cutoff`` or -1."""
        return int(np.searchsorted(self.available, np.datetime64(cutoff.tz_convert("UTC").tz_localize(None)), side="right")) - 1


def _series(df: pd.DataFrame, value_col: str, quality_col: str | None = None) -> Series:
    df = df.assign(_ev=to_utc(df["event_timestamp_utc"]), _av=to_utc(df["available_at_utc"])).sort_values("_ev")
    ev = df["_ev"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    av = df["_av"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    if np.any(np.diff(ev.astype("int64")) <= 0):
        raise ValueError("non-monotonic or duplicate event timestamps")
    if np.any(av < ev):
        raise ValueError("available_at_utc precedes event_timestamp_utc")
    quality = df[quality_col].to_numpy() if quality_col else np.array(["ok"] * len(df))
    return Series(ev, av, df[value_col].to_numpy(dtype=float), quality)


@dataclass(frozen=True)
class MarketData:
    raw: dict[str, Series]
    total_return: dict[str, Series]
    fx: Series
    rates: pd.DataFrame  # rate_date, event_timestamp_utc, available_at_utc, annual_rate
    actions: pd.DataFrame
    max_fx_stale: pd.Timedelta
    cache: dict = field(default_factory=dict, compare=False, repr=False)  # derived views of this immutable data (see fitstate.full_daily_view)

    def raw_mark(self, asset: str, cutoff: pd.Timestamp) -> Decimal:
        s = self.raw[asset]
        i = s.asof_index(cutoff)
        if i < 0:
            raise NoDataError(f"{asset}: no raw price available by {cutoff}")
        return Decimal(repr(float(s.value[i])))

    def raw_close_at(self, asset: str, event_ts: pd.Timestamp) -> Decimal:
        """The raw close whose own event timestamp is ``event_ts`` (the executable fill price)."""
        s = self.raw[asset]
        key = np.datetime64(event_ts.tz_convert("UTC").tz_localize(None))
        i = int(np.searchsorted(s.event, key))
        if i >= len(s.event) or s.event[i] != key:
            raise NoDataError(f"{asset}: no executable close at {event_ts}")
        if s.quality[i] != "ok":
            raise NoDataError(f"{asset}: close at {event_ts} flagged {s.quality[i]}")
        return Decimal(repr(float(s.value[i])))

    def tradable(self, asset: str, cutoff: pd.Timestamp) -> bool:
        """True when the latest raw observation available by ``cutoff`` exists and is not flagged (e.g. liquidity_invalid)."""
        s = self.raw[asset]
        i = s.asof_index(cutoff)
        return bool(i >= 0 and s.quality[i] == "ok")

    def fx_mark(self, cutoff: pd.Timestamp) -> Decimal:
        i = self.fx.asof_index(cutoff)
        if i < 0:
            raise NoDataError(f"no FX quote available by {cutoff}")
        age = cutoff.tz_convert("UTC").tz_localize(None) - pd.Timestamp(self.fx.event[i])
        if age > self.max_fx_stale:
            raise StaleDataError(f"FX quote age {age} exceeds {self.max_fx_stale} at {cutoff}")
        return Decimal(repr(float(self.fx.value[i])))


def load_market_data(cfg: Config, root: Path) -> MarketData:
    d = cfg.data
    raw = pd.read_parquet(root / d.prices_raw)
    tr = pd.read_parquet(root / d.prices_total_return)
    fx = pd.read_parquet(root / d.fx)
    fx = fx.assign(event_timestamp_utc=fx["event_timestamp_utc"])
    max_stale = pd.Timedelta(hours=cfg.availability.max_fx_stale_hours)
    return MarketData(
        raw={a: _series(g, "value", "quality_flag") for a, g in raw.groupby("instrument_id")},
        total_return={a: _series(g, "value", "quality_flag") for a, g in tr.groupby("instrument_id")},
        fx=_series(fx, "value"),
        rates=pd.read_parquet(root / d.rates),
        actions=pd.read_parquet(root / d.corporate_actions),
        max_fx_stale=max_stale,
    )


def load_initial_holdings(cfg: Config, root: Path) -> tuple[Decimal, dict[str, int], pd.Timestamp]:
    h = pd.read_csv(root / cfg.data.initial_holdings, dtype=str, keep_default_na=False)
    if h.asset_id.duplicated().any():
        raise ValueError("duplicate asset in initial holdings")
    known = {a.asset_id for a in (cfg.universe.assets or [])} | {cfg.universe.cash_asset_id}
    if not set(h.asset_id) <= known:
        raise ValueError(f"unknown assets in initial holdings: {sorted(set(h.asset_id) - known)}")
    as_of = pd.Timestamp(h.as_of_utc.iloc[0])
    if (pd.to_datetime(h.as_of_utc, utc=True) != as_of).any():
        raise ValueError("initial holdings must share one as_of_utc")
    cash_rows = h[h.asset_id == cfg.universe.cash_asset_id]
    if len(cash_rows) != 1 or Decimal(cash_rows.cash_amount.iloc[0]) < 0:
        raise ValueError("exactly one non-negative CASH_KRW row required")
    qty = {r.asset_id: int(r.quantity) for r in h.itertuples() if r.asset_id != cfg.universe.cash_asset_id}
    if any(q < 0 for q in qty.values()):
        raise ValueError("negative initial quantity")
    return Decimal(cash_rows.cash_amount.iloc[0]), qty, as_of
