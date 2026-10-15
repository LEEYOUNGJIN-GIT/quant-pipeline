"""Canonical valuation clock (spec 5.1.1). Every XNYS anchor is a valuation row; XKRX closures carry the latest raw mark."""
from __future__ import annotations

import pandas as pd

from .calendars import Calendars, RowMap, fmt
from .config import Config


class ValuationStaleBreach(RuntimeError):
    """Consecutive XNYS anchors since the last XKRX raw close exceed ``valuation.max_krx_stale_sessions`` (halt)."""


def valuation_table(cfg: Config, cals: Calendars, rm: RowMap, xkrx_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per anchor: owner block, ``krx_stale_count`` and ``valuation_stale``. Halts on a limit breach."""
    limit = cfg.valuation.max_krx_stale_sessions
    krx = set(xkrx_sessions)
    run = 0
    rows = []
    for s in rm.anchors:
        run = 0 if s in krx else run + 1
        if run > limit:
            raise ValuationStaleBreach(f"{s.date()}: XKRX raw mark stale for {run} anchors (limit {limit})")
        rows.append({"session_date": s.strftime("%Y-%m-%d"), "valuation_utc": fmt(cals.close("XNYS", s)),
                     "owner": rm.valuation_owner(s), "krx_stale_count": run, "valuation_stale": run > 0})
    return pd.DataFrame(rows)


def owned_interval_counts(table: pd.DataFrame) -> dict[str, int]:
    """Scored valuation intervals per block: each owned anchor ends exactly one interval."""
    owned = table[table.owner != ""]
    return owned.groupby("owner").size().to_dict()


def max_stale_run(xnys: pd.DatetimeIndex, xkrx: pd.DatetimeIndex) -> int:
    krx = set(xkrx)
    run = best = 0
    for s in xnys:
        run = 0 if s in krx else run + 1
        best = max(best, run)
    return best
