"""Hand-made responses in the shape Yahoo (yfinance) and ECOS return, for offline tests of the live adapters (never real data).

The world is generated from as-traded prices, then expressed the way Yahoo expresses it: ``Close`` adjusted for splits over the whole
history, dividends per share in the same adjusted units, one bar per official session indexed at local midnight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pipeline.calendars import Calendars
from quant_pipeline.yfinance_adapter import YahooHistory

CALS = Calendars.load(["XKRX", "XNYS"])
TZ = {"XKRX": "Asia/Seoul", "XNYS": "America/New_York"}
CCY = {"XKRX": "KRW", "XNYS": "USD"}


def asset_ids() -> list[tuple[str, str]]:
    return [(f"{c}.KS", "XKRX") for c in ("005930", "000660", "005380", "035420", "069500")] + [(s, "XNYS") for s in ("SPY", "IEF", "TLT", "GLD")]


def yahoo_frame(symbol: str, exchange: str, start: str, end: str, *, split_on: str | None = None, split_ratio: float = 2.0,
                dividend_every: int = 63, dividend_yield: float = 0.005, seed: int = 0, noise_bars: bool = True) -> YahooHistory:
    rng = np.random.default_rng(sum(map(ord, symbol)) + seed)
    sessions = CALS.sessions(exchange, start, end)
    n = len(sessions)
    level = 50_000.0 if exchange == "XKRX" else 100.0
    raw = level * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    split_pos = int(sessions.get_loc(pd.Timestamp(split_on))) if split_on else None
    if split_pos is not None:
        raw[split_pos:] = raw[split_pos:] / split_ratio  # the as-traded price drops by the split ratio on the split day
    divs = np.zeros(n)
    for i in range(dividend_every, n, dividend_every):
        divs[i] = round(raw[i - 1] * dividend_yield, 2 if exchange == "XNYS" else 0)  # as-traded cash per share at the ex date
    after = np.ones(n)
    if split_pos is not None:
        after[:split_pos] = split_ratio  # splits strictly after bar i
    close_adj, div_adj = raw / after, divs / after
    splits = np.zeros(n)
    if split_pos is not None:
        splits[split_pos] = split_ratio
    df = pd.DataFrame({"Open": close_adj, "High": close_adj, "Low": close_adj, "Close": close_adj, "Adj Close": close_adj,
                       "Volume": np.full(n, 1000.0), "Dividends": div_adj, "Stock Splits": splits},
                      index=pd.DatetimeIndex(sessions).tz_localize(TZ[exchange]))
    if noise_bars:
        df.loc[df.index[7], "Volume"] = 0.0  # a session with no trading (set before the extra bar is inserted)
        saturday = next(d for d in pd.date_range(sessions[0], periods=7) if d.weekday() == 5)  # a bar on a day that is not a session
        df.loc[pd.Timestamp(saturday).tz_localize(TZ[exchange])] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        df = df.sort_index()
    return YahooHistory(df, CCY[exchange], TZ[exchange])


class FakeYahoo:
    def __init__(self, start: str, end: str, split: dict[str, str] | None = None):
        self.start, self.end, self.split = start, end, split or {}
        self.exchanges = dict(asset_ids())
        self.calls: list[str] = []

    def history(self, symbol: str, start: str, end: str) -> YahooHistory:
        self.calls.append(symbol)
        if symbol == "KRW=X":
            days = pd.bdate_range(start, end)
            rng = np.random.default_rng(7)
            vals = 1300.0 * np.exp(np.cumsum(rng.normal(0, 0.003, len(days))))
            df = pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals, "Adj Close": vals, "Volume": 0.0, "Dividends": 0.0,
                               "Stock Splits": 0.0}, index=pd.DatetimeIndex(days).tz_localize("Europe/London"))
            return YahooHistory(df, "KRW", "Europe/London")
        return yahoo_frame(symbol, self.exchanges[symbol], start, end, split_on=self.split.get(symbol))


class FakeEcos:
    def __init__(self, pct: float = 3.5):
        self.pct = pct

    def series(self, stat_code: str, item_code: str, start: str, end: str) -> list[dict[str, str]]:
        days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))  # includes weekdays that are XKRX holidays
        return [{"TIME": d.strftime("%Y%m%d"), "DATA_VALUE": f"{self.pct:.2f}"} for d in days]
