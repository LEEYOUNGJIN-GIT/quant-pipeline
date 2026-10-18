"""Canonical synchronized return panel (spec 5.1) and weekly common-cutoff sensitivity panel (spec 5.3).

Causality is by ``available_at_utc``. A return exists only for a newly observed level; a closed market yields
null (NaN), never 0.0. Rows are built independently of later rows, so a future observation cannot change an earlier row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendars import Calendars, fmt, parse_delay, utc
from .data import Series

PANEL_AGE_LIMIT_WEEKLY = pd.Timedelta(days=7)


def _row_cols(asset: str) -> list[str]:
    return [f"{asset}_level", f"{asset}_updated", f"{asset}_return", f"{asset}_age_hours"]


def _asof_idx(series: Series, cutoffs: list[pd.Timestamp]) -> np.ndarray:
    return np.array([series.asof_index(c) for c in cutoffs], dtype=int)


def _age_hours(series: Series, idx: int, cutoff: pd.Timestamp) -> float | None:
    if idx < 0:
        return None
    return float((np.datetime64(cutoff.tz_convert("UTC").tz_localize(None)) - series.event[idx]) / np.timedelta64(1, "h"))


def daily_panel(levels: dict[str, Series], cals: Calendars, anchors: pd.DatetimeIndex, delay: str,
                anchor_calendar: str = "XNYS") -> pd.DataFrame:
    """One row per anchor session. ``cutoff = anchor close + publication delay`` (XNYS baseline, XKRX alternate)."""
    d = parse_delay(delay)
    cutoffs = [cals.close(anchor_calendar, s) + d for s in anchors]
    cols: dict[str, list] = {"kind": ["daily"] * len(cutoffs), "cutoff_utc": [fmt(c) for c in cutoffs],
                             "session_date": [s.strftime("%Y-%m-%d") for s in anchors]}
    complete = np.ones(len(cutoffs), dtype=bool)
    for asset, s in levels.items():
        idx = _asof_idx(s, cutoffs)
        prev = np.concatenate([[-1], idx[:-1]])
        updated = (idx >= 0) & (idx != prev)
        has_prev = prev >= 0
        ret = np.full(len(idx), np.nan)
        ok = updated & has_prev
        ret[ok] = s.value[idx[ok]] / s.value[prev[ok]] - 1.0
        cols[f"{asset}_level"] = [float(s.value[i]) if i >= 0 else np.nan for i in idx]
        cols[f"{asset}_updated"] = updated.tolist()
        cols[f"{asset}_return"] = ret.tolist()
        cols[f"{asset}_age_hours"] = [_age_hours(s, int(i), c) for i, c in zip(idx, cutoffs, strict=True)]
        complete &= ok
    cols["complete_case"] = complete.tolist()
    return pd.DataFrame(cols)


def weekly_panel(levels: dict[str, Series], first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Friday 23:59:59Z common cutoffs; a return needs an advanced source event and age <= 7 days."""
    fridays = pd.date_range(first - pd.Timedelta(days=first.weekday()) + pd.Timedelta(days=4), last, freq="7D")
    cutoffs = [utc(f.strftime("%Y-%m-%d") + "T23:59:59") for f in fridays]
    cols: dict[str, list] = {"kind": ["weekly"] * len(cutoffs), "cutoff_utc": [fmt(c) for c in cutoffs],
                             "session_date": [f.strftime("%Y-%m-%d") for f in fridays]}
    complete = np.ones(len(cutoffs), dtype=bool)
    for asset, s in levels.items():
        idx = _asof_idx(s, cutoffs)
        age = [_age_hours(s, int(i), c) for i, c in zip(idx, cutoffs, strict=True)]
        fresh = np.array([a is not None and a <= PANEL_AGE_LIMIT_WEEKLY.total_seconds() / 3600 for a in age])
        # the comparison level is the previous week's as-of observation, whether or not it was fresh
        prev = np.full(len(idx), -1, dtype=int)
        last_seen = -1
        for k, i in enumerate(idx):
            prev[k] = last_seen
            if i >= 0:
                last_seen = int(i)
        updated = fresh & (idx >= 0) & (idx != prev)
        ok = updated & (prev >= 0)
        ret = np.full(len(idx), np.nan)
        ret[ok] = s.value[idx[ok]] / s.value[prev[ok]] - 1.0
        cols[f"{asset}_level"] = [float(s.value[i]) if i >= 0 else np.nan for i in idx]
        cols[f"{asset}_updated"] = updated.tolist()
        cols[f"{asset}_return"] = ret.tolist()
        cols[f"{asset}_age_hours"] = age
        complete &= ok
    cols["complete_case"] = complete.tolist()
    return pd.DataFrame(cols)


def panel_tables(levels: dict[str, Series], cals: Calendars, sessions_xnys: pd.DatetimeIndex, delay: str) -> pd.DataFrame:
    """Daily rows for every XNYS session in the window followed by weekly rows (the oracle's table layout)."""
    daily = daily_panel(levels, cals, sessions_xnys, delay)
    weekly = weekly_panel(levels, sessions_xnys[0], sessions_xnys[-1])
    return pd.concat([daily, weekly], ignore_index=True)
