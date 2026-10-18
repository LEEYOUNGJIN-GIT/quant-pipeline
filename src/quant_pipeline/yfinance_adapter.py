"""Yahoo Finance (``yfinance``) mapping to the point-in-time snapshot schema (spec 4.6; OQ-036, OQ-050).

Everything here is a pure transformation of what the service returns, so it is tested offline with hand-made responses; only
``RealYahooClient`` touches the network and it is imported lazily (``research-online`` extra).

What the mapping must undo or assume, because Yahoo does not match the project's conventions:

* **Raw prices.** Yahoo's ``Close`` is adjusted for splits over the whole history. The project's raw series is the price as traded (the fixture's
  raw price jumps on a split day while the total-return index is continuous), and the engine rescales quantities at the split. The split
  adjustment is therefore undone with the split table the same response carries: ``raw_t = close_t * prod(ratio of splits after t)``.
* **Dividends.** Yahoo reports them per share in today's (split-adjusted) units; the project wants the cash per share held at the ex date in
  as-traded units, so they are rescaled the same way.
* **Total return.** Built here from the response under the project's convention (gross dividends reinvested at the ex-date close, OQ-035):
  ``TR_t = TR_(t-1) * (close_t + dividend_t) / close_(t-1)`` in split-adjusted units. Yahoo's ``Adj Close`` reinvests at the previous close
  and differs by about ``d * (return + d)`` per dividend, which exceeds the 5 bp per-event limit for ordinary dividends, so it is not used as the
  index. Level 2 reconciliation is therefore a consistency check of this construction, not an independent one; the resulting snapshot says so.
* **Timing.** Yahoo gives no publication time. A daily bar is stamped at the official session close and made available one configured
  publication delay later; a dividend or split is taken as known at the close of the session before its ex session (declaration dates are not provided; real announcements are earlier, so this is the safer side) and paid at the ex close.
* **No vintages.** Yahoo does not keep point-in-time revisions, so ``source_revision`` is a constant and the data may differ on a later fetch.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .calendars import Calendars, fmt, parse_delay
from .collector import (
    ACTION_COLUMNS,
    FX_COLUMNS,
    PRICE_COLUMNS,
    Provider,
    ProviderError,
    ProviderResponse,
    Request,
    SchemaDrift,
    SymbolNotFound,
    TransientError,
)

SOURCE_ID = "yahoo_finance"
SOURCE_REVISION = "unversioned"
FX_SYMBOLS = {"USDKRW": "KRW=X"}  # KRW per USD, quoted by Yahoo as KRW=X
FX_BAR_TIMEZONE = "Europe/London"  # the daily FX bar is the London calendar day: its close is the end of that day (checked against the response metadata)
REQUIRED_COLUMNS = ("Close", "Dividends", "Stock Splits", "Volume")
ROUNDING = {"KRW": 2, "USD": 4}
MAX_SPLIT_DENOMINATOR = 1000


@dataclass(frozen=True)
class YahooHistory:
    frame: pd.DataFrame  # index: bar dates; columns as returned by ``Ticker.history(auto_adjust=False, actions=True)``
    currency: str | None
    timezone: str | None


class YahooClient(Protocol):
    def history(self, symbol: str, start: str, end: str) -> YahooHistory: ...


class RealYahooClient:
    """The only network-touching piece. ``end`` is inclusive here (yfinance's own ``end`` is exclusive)."""

    def history(self, symbol: str, start: str, end: str) -> YahooHistory:
        try:
            import yfinance as yf  # optional dependency (extra ``research-online``)
        except ImportError as e:
            raise ProviderError("yfinance is not installed; install the 'research-online' extra") from e
        stop = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(symbol)
        try:
            frame = ticker.history(start=start, end=stop, interval="1d", auto_adjust=False, actions=True, raise_errors=True)
            meta = ticker.history_metadata or {}
        except Exception as e:  # noqa: BLE001 - the service raises assorted library errors; classify without leaking request details
            text = str(e).lower()
            if "delisted" in text or "no data found" in text or "not found" in text:
                raise SymbolNotFound(symbol) from None
            raise TransientError(f"yahoo request failed: {type(e).__name__}") from None
        return YahooHistory(frame, meta.get("currency"), meta.get("exchangeTimezoneName"))


# ----------------------------------------------------------------------------------------------------------------- pure mapping
def _bar_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # the wall-clock date of the bar in the exchange's own timezone
    return idx.normalize()


def _check_columns(symbol: str, frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaDrift(f"{symbol}: the response lacks columns {missing}")


def split_factors(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """For each bar, the product of the split ratios that take effect strictly after it (1.0 after the last split)."""
    ratio = np.where(frame["Stock Splits"].fillna(0.0).to_numpy(float) > 0, frame["Stock Splits"].fillna(0.0).to_numpy(float), 1.0)
    after = np.ones(len(dates))
    running = 1.0
    for i in range(len(dates) - 1, -1, -1):
        after[i] = running
        running *= ratio[i]
    return after


def clean_frame(symbol: str, frame: pd.DataFrame, calendar: str, cals: Calendars) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep official sessions with a usable close; report what was dropped instead of hiding it."""
    _check_columns(symbol, frame)
    if frame.empty:
        raise SymbolNotFound(symbol)
    dates = _bar_dates(frame)
    df = frame.copy()
    df.index = dates
    if df.index.has_duplicates:
        raise SchemaDrift(f"{symbol}: duplicate bar dates")
    df = df.sort_index()
    sessions = cals.sessions(calendar, df.index[0], df.index[-1])
    in_session = df.index.isin(sessions)
    usable = df["Close"].notna() & (df["Close"] > 0)
    dropped = {"non_session_rows": int((~in_session).sum()), "unusable_close_rows": int((in_session & ~usable).sum())}
    return df[in_session & usable], dropped


def price_frames(symbol: str, history: YahooHistory, *, exchange: str, currency: str, cals: Calendars, price_delay: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(raw close, total-return index, diagnostics) for one instrument, in the snapshot's canonical column order."""
    df, dropped = clean_frame(symbol, history.frame, exchange, cals)
    if history.currency and history.currency != currency:
        raise SchemaDrift(f"{symbol}: Yahoo quotes in {history.currency}, the universe record says {currency}")
    delay = parse_delay(price_delay)
    dates = df.index
    closes = [cals.close(exchange, d) for d in dates]
    split_adj = split_factors(df, dates)
    adj_close = df["Close"].to_numpy(float)
    raw = np.round(adj_close * split_adj, ROUNDING[currency])
    div_adj = df["Dividends"].fillna(0.0).to_numpy(float)
    prev = np.concatenate([[np.nan], adj_close[:-1]])
    growth = np.where(np.isnan(prev), 1.0, (adj_close + div_adj) / prev)
    tr = 100.0 * np.cumprod(growth)
    zero_volume = (df["Volume"].fillna(0.0).to_numpy(float) <= 0)
    sessions = [d.strftime("%Y-%m-%d") for d in dates]

    def frame(values: np.ndarray, field: str, unit: str, flags: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "instrument_id": symbol, "source_timestamp": [fmt(c) for c in closes], "event_timestamp_utc": [fmt(c) for c in closes],
            "available_at_utc": [fmt(c + delay) for c in closes], "exchange_session": sessions, "field": field, "value": values.astype(float),
            "currency": currency, "unit": unit, "source_id": SOURCE_ID, "source_revision": SOURCE_REVISION, "is_final": True, "is_stale": False,
            "quality_flag": flags})[PRICE_COLUMNS]

    raw_frame = frame(raw, "close", f"{currency}/share", ["liquidity_invalid" if z else "ok" for z in zero_volume])
    tr_frame = frame(tr, "total_return_index", "index", ["ok"] * len(dates))
    diag = {**dropped, "bars": len(dates), "dividends": int((div_adj > 0).sum()), "splits": int((df["Stock Splits"].fillna(0.0) > 0).sum()),
            "zero_volume_bars": int(zero_volume.sum()), "first_session": sessions[0], "last_session": sessions[-1]}
    return raw_frame, tr_frame, diag


def action_frame(symbol: str, history: YahooHistory, *, exchange: str, currency: str, cals: Calendars) -> pd.DataFrame:
    df, _ = clean_frame(symbol, history.frame, exchange, cals)
    split_adj = split_factors(df, df.index)
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(df.index):
        ex_ts = fmt(cals.close(exchange, d))
        day = d.strftime("%Y-%m-%d")
        known = fmt(cals.close(exchange, cals.cals[exchange].session_offset(d, -1)))
        # Declaration and pay dates are not provided. The snapshot rule needs the action known strictly before the ex timestamp, so it is taken
        # as known at the close of the previous session (announcements normally come weeks earlier, so this is the later, safer choice) and paid
        # at the ex close.
        base = {"instrument_id": symbol, "declared_at_utc": known, "available_at_utc": known, "ex_session_date": day, "ex_timestamp_utc": ex_ts,
                "pay_session_date": day, "pay_timestamp_utc": ex_ts}
        dividend = float(df["Dividends"].iloc[i]) if pd.notna(df["Dividends"].iloc[i]) else 0.0
        if dividend > 0:
            rows.append({**base, "action_id": f"{symbol}:cash_dividend:{day}", "action_type": "cash_dividend", "ratio_num": np.nan, "ratio_den": np.nan,
                         "cash_amount": round(dividend * split_adj[i], ROUNDING[currency] + 2), "cash_currency": currency})
        ratio = float(df["Stock Splits"].iloc[i]) if pd.notna(df["Stock Splits"].iloc[i]) else 0.0
        if ratio > 0 and ratio != 1.0:
            fr = Fraction(ratio).limit_denominator(MAX_SPLIT_DENOMINATOR)
            if abs(float(fr) - ratio) > 1e-9 * ratio:
                raise SchemaDrift(f"{symbol}: split ratio {ratio} on {day} is not a rational with denominator <= {MAX_SPLIT_DENOMINATOR}")
            rows.append({**base, "action_id": f"{symbol}:split:{day}", "action_type": "split", "ratio_num": float(fr.numerator),
                         "ratio_den": float(fr.denominator), "cash_amount": np.nan, "cash_currency": np.nan})
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def fx_frame(pair: str, history: YahooHistory, *, fx_delay: str, end: str | None = None) -> pd.DataFrame:
    """One quote per weekday, stamped at the end of its London day (the bar's close), available one publication delay later.

    A bar that closes after the end of the requested range is not emitted: its value was not yet known inside the range.
    """
    _check_columns(pair, history.frame)
    if history.timezone and history.timezone != FX_BAR_TIMEZONE:
        raise SchemaDrift(f"{pair}: the FX bar timezone is {history.timezone}, expected {FX_BAR_TIMEZONE}")
    dates = _bar_dates(history.frame)
    df = history.frame.copy()
    df.index = dates
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    if df.empty:
        raise SymbolNotFound(pair)
    ends = [pd.Timestamp(d.strftime("%Y-%m-%d"), tz=FX_BAR_TIMEZONE).tz_convert("UTC") + pd.Timedelta(days=1) for d in df.index]
    if end is not None:
        keep = [e.strftime("%Y-%m-%d") <= end for e in ends]
        ends, df = [e for e, k in zip(ends, keep, strict=True) if k], df[keep]
        if df.empty:
            raise SymbolNotFound(pair)
    delay = parse_delay(fx_delay)
    return pd.DataFrame({"pair_id": pair, "event_timestamp_utc": [fmt(e) for e in ends], "available_at_utc": [fmt(e + delay) for e in ends],
                         "value": df["Close"].to_numpy(float), "quote_convention": "KRW_per_USD", "source_id": SOURCE_ID})[FX_COLUMNS]


# ----------------------------------------------------------------------------------------------------------------- provider
@dataclass(frozen=True)
class AssetBinding:
    exchange: str
    currency: str


class YahooProvider(Provider):
    """Prices, total return, corporate actions, metadata and FX from Yahoo. Rates are not offered (see the ECOS adapter)."""

    name = "yfinance"
    adapter_version = "1"
    license = ("Yahoo Finance data through the unofficial yfinance client: for personal, non-commercial research use under Yahoo's terms; "
               "not redistributable; no service-level or accuracy guarantee (user-attested)")

    def __init__(self, client: YahooClient, assets: dict[str, AssetBinding], cals: Calendars, *, price_delay: str, fx_delay: str):
        self.client, self.assets, self.cals, self.price_delay, self.fx_delay = client, assets, cals, price_delay, fx_delay
        self.diagnostics: dict[str, dict[str, Any]] = {}
        self._memo: dict[tuple[str, str, str], YahooHistory] = {}
        self._lock = threading.Lock()  # the collector fetches concurrently: without it two requests for one symbol would both download

    def _history(self, symbol: str, start: str, end: str) -> YahooHistory:
        key = (symbol, start, end)
        with self._lock:
            if key not in self._memo:  # raw, total return, actions and metadata share one download per symbol
                self._memo[key] = self.client.history(symbol, start, end)
            return self._memo[key]

    def _binding(self, symbol: str) -> AssetBinding:
        if symbol not in self.assets:
            raise SymbolNotFound(symbol)
        return self.assets[symbol]

    def fetch_metadata(self, req: Request) -> ProviderResponse:
        b = self._binding(req.symbol)
        h = self._history(req.symbol, req.start, req.end)
        return ProviderResponse(pd.DataFrame({"symbol": [req.symbol], "currency": [h.currency or b.currency]}), {"source": SOURCE_ID})

    def _prices(self, req: Request) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        b = self._binding(req.symbol)
        raw, tr, diag = price_frames(req.symbol, self._history(req.symbol, req.start, req.end), exchange=b.exchange, currency=b.currency,
                                     cals=self.cals, price_delay=self.price_delay)
        self.diagnostics[req.symbol] = diag
        return raw, tr, diag

    def fetch_raw_prices(self, req: Request) -> ProviderResponse:
        raw, _, diag = self._prices(req)
        return ProviderResponse(raw, {"source": SOURCE_ID, **{k: v for k, v in diag.items() if isinstance(v, int)}}, SOURCE_REVISION)

    def fetch_total_return(self, req: Request) -> ProviderResponse:
        _, tr, _ = self._prices(req)
        return ProviderResponse(tr, {"source": SOURCE_ID, "construction": "derived_from_raw_and_dividends"}, SOURCE_REVISION)

    def fetch_corporate_actions(self, req: Request) -> ProviderResponse:
        b = self._binding(req.symbol)
        return ProviderResponse(action_frame(req.symbol, self._history(req.symbol, req.start, req.end), exchange=b.exchange, currency=b.currency,
                                             cals=self.cals), {"source": SOURCE_ID}, SOURCE_REVISION)

    def fetch_fx(self, req: Request) -> ProviderResponse:
        if req.symbol not in FX_SYMBOLS:
            raise SymbolNotFound(req.symbol)
        # the response carries one extra leading day so the first requested weekday has its own bar
        h = self._history(FX_SYMBOLS[req.symbol], req.start, req.end)
        return ProviderResponse(fx_frame(req.symbol, h, fx_delay=self.fx_delay, end=req.end), {"source": SOURCE_ID}, SOURCE_REVISION)
