"""Data-quality report of a collected snapshot, with pass/fail criteria fixed here before any real data is looked at (OQ-050).

It reports aggregate statistics only (counts, dates, extremes), never raw series, so it is safe to print in public logs. A criterion that fails
means the snapshot must not feed a research run; nothing here relaxes a threshold to make a snapshot pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calendars import Calendars

MIN_SESSION_COVERAGE = 0.99  # share of official sessions with a usable close
MAX_CONSECUTIVE_MISSING = 5  # sessions; longer than the 2-session staleness limit would already break the panel
MAX_ABS_DAILY_RETURN = 0.40  # outside split days, a larger as-traded move is treated as a data error until looked at
MIN_DIVIDENDS_PER_YEAR_FOR_PAYERS = 0.5  # instruments flagged as dividend payers must show roughly this many events a year
MAX_FX_AGE_HOURS = 72.0  # availability.max_fx_stale_hours
MAX_RATE_GAP_BUSINESS_DAYS = 5


def _max_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def instrument_report(symbol: str, exchange: str, raw: pd.DataFrame, actions: pd.DataFrame, cals: Calendars, expects_dividends: bool) -> dict[str, Any]:
    sessions = cals.sessions(exchange, raw.exchange_session.min(), raw.exchange_session.max())
    have = pd.DatetimeIndex(pd.to_datetime(raw.exchange_session))
    present = sessions.isin(have)
    coverage = float(present.mean())
    px = raw.sort_values("exchange_session").set_index("exchange_session").value
    ret = px.pct_change().dropna()
    split_days = set(actions[(actions.instrument_id == symbol) & (actions.action_type == "split")].ex_session_date)
    plain = ret[~ret.index.isin(split_days)]
    years = max((pd.Timestamp(px.index[-1]) - pd.Timestamp(px.index[0])).days / 365.25, 1e-9)
    divs = actions[(actions.instrument_id == symbol) & (actions.action_type == "cash_dividend")]
    per_year = len(divs) / years
    out = {
        "symbol": symbol, "exchange": exchange, "bars": len(raw), "first": str(px.index[0]), "last": str(px.index[-1]),
        "session_coverage": round(coverage, 5), "max_consecutive_missing_sessions": _max_run(~present),
        "zero_volume_sessions": int((raw.quality_flag != "ok").sum()),
        # official session dates (calendar facts, not provider data) without a usable bar, to tell provider gaps from calendar mismatches
        "missing_sessions_by_year": {str(y): int(n) for y, n in pd.Series(sessions[~present].year).value_counts().sort_index().items()},
        "missing_sessions_first": [str(d.date()) for d in sessions[~present][:60]],
        "dividends": len(divs), "dividends_per_year": round(per_year, 3), "splits": len(split_days),
        "max_abs_daily_return_excluding_splits": round(float(plain.abs().max()), 4) if len(plain) else 0.0,
    }
    fails = []
    if coverage < MIN_SESSION_COVERAGE:
        fails.append(f"session coverage {coverage:.4f} < {MIN_SESSION_COVERAGE}")
    if out["max_consecutive_missing_sessions"] > MAX_CONSECUTIVE_MISSING:
        fails.append(f"{out['max_consecutive_missing_sessions']} consecutive missing sessions")
    if out["max_abs_daily_return_excluding_splits"] > MAX_ABS_DAILY_RETURN:
        fails.append(f"daily move {out['max_abs_daily_return_excluding_splits']:.2f} outside split days")
    if expects_dividends and per_year < MIN_DIVIDENDS_PER_YEAR_FOR_PAYERS:
        fails.append(f"{per_year:.2f} dividends a year for an instrument expected to pay (the provider's dividend history is probably incomplete)")
    out["failures"] = fails
    return out


def fx_report(fx: pd.DataFrame, cals: Calendars, max_age_hours: float = MAX_FX_AGE_HOURS) -> dict[str, Any]:
    """The pipeline's own rule: at every official XKRX and XNYS close the latest available quote must be at most ``max_age_hours`` old."""
    ev = pd.to_datetime(fx.event_timestamp_utc, utc=True).to_numpy("datetime64[ns]")
    av = pd.to_datetime(fx.available_at_utc, utc=True).to_numpy("datetime64[ns]")
    order = np.argsort(av)
    ev, av = ev[order], av[order]
    first_av, last_av = pd.Timestamp(av[0]).tz_localize("UTC"), pd.Timestamp(av[-1]).tz_localize("UTC")
    ages: list[float] = []
    for name in ("XKRX", "XNYS"):
        for s in cals.sessions(name, first_av.tz_localize(None).normalize(), last_av.tz_localize(None).normalize()):
            close = cals.close(name, s)
            if close < first_av or close > last_av:
                continue
            i = int(np.searchsorted(av, np.datetime64(close.tz_localize(None)), side="right")) - 1
            ages.append((np.datetime64(close.tz_localize(None)) - ev[i]) / np.timedelta64(1, "h"))
    worst = float(max(ages)) if ages else 0.0
    out = {"quotes": len(fx), "first": str(first_av)[:10], "last": str(last_av)[:10], "closes_checked": len(ages), "max_age_at_close_hours": round(worst, 2),
           "min": round(float(fx.value.min()), 2), "max": round(float(fx.value.max()), 2)}
    out["failures"] = [f"FX quote {worst:.1f} h old at a close, limit {max_age_hours}"] if worst > max_age_hours else []
    return out


def rates_report(rates: pd.DataFrame, cals: Calendars) -> dict[str, Any]:
    dates = pd.DatetimeIndex(pd.to_datetime(rates.rate_date)).sort_values()
    sessions = cals.sessions("XKRX", dates[0], dates[-1])
    missing = ~sessions.isin(dates)
    out = {"rows": len(rates), "first": str(dates[0].date()), "last": str(dates[-1].date()), "max_consecutive_missing_sessions": _max_run(missing),
           "min_pct": round(float(rates.annual_rate.min()) * 100, 3), "max_pct": round(float(rates.annual_rate.max()) * 100, 3)}
    out["failures"] = [f"{out['max_consecutive_missing_sessions']} consecutive XKRX sessions without a rate"] if out["max_consecutive_missing_sessions"] > MAX_RATE_GAP_BUSINESS_DAYS else []
    return out


def snapshot_report(snapshot: Path, universe: list[dict[str, Any]], dividend_payers: frozenset[str] = frozenset(), cals: Calendars | None = None) -> dict[str, Any]:
    cals = cals or Calendars.load(["XKRX", "XNYS"])
    raw = pd.read_parquet(snapshot / "prices_raw.parquet")
    actions = pd.read_parquet(snapshot / "corporate_actions.parquet")
    instruments = []
    for rec in universe:
        s = rec["asset_id"]
        instruments.append(instrument_report(s, rec["exchange"], raw[raw.instrument_id == s], actions, cals, s in dividend_payers))
    fx, rates = fx_report(pd.read_parquet(snapshot / "fx_krw_per_usd.parquet"), cals), rates_report(pd.read_parquet(snapshot / "rates_krw_rfr.parquet"), cals)
    failures = [f"{i['symbol']}: {m}" for i in instruments for m in i["failures"]] + [f"fx: {m}" for m in fx["failures"]] + [f"rates: {m}" for m in rates["failures"]]
    return {"snapshot": snapshot.name, "instruments": instruments, "fx": fx, "rates": rates, "failures": failures, "passed": not failures}


def write_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
