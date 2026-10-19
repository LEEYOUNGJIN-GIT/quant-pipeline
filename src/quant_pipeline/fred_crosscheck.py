"""Cross-check of the snapshot's USD/KRW series against FRED ``DEXKOUS`` (Federal Reserve H.10 noon buying rate, KRW per USD).

This is a closeness check, not an equality check: Yahoo's ``KRW=X`` bar is stamped at the end of its London day while H.10 is a New York noon
quote, so the two differ by the intraday move. FRED is never an input to the research (the series is published with a lag and is not point in time);
the result is a separate report and changes neither the snapshot nor any hash. The pass criteria below were fixed before any real comparison was
made and are not to be loosened to make a snapshot pass. The API key is in the query string, so no URL or request detail may reach a message.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .collector import ProviderError, RateLimited, SchemaDrift, TransientError
from .yfinance_adapter import FX_BAR_TIMEZONE

SERIES_ID = "DEXKOUS"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
MIN_MATCHED_SHARE = 0.90  # share of FRED dates inside the snapshot's FX range that have a snapshot quote for the same date
MAX_MEDIAN_ABS_REL_DIFF = 0.005
MAX_P99_ABS_REL_DIFF = 0.02
MAX_ABS_REL_DIFF = 0.05


class FredClient(Protocol):
    def observations(self, series_id: str, start: str, end: str) -> list[dict[str, str]]: ...


class RealFredClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self._key = api_key or os.environ.get("FRED_API_KEY", "")
        self.timeout = timeout
        if not self._key:
            raise ProviderError("FRED_API_KEY is not set")

    def observations(self, series_id: str, start: str, end: str) -> list[dict[str, str]]:
        query = urllib.parse.urlencode({"series_id": series_id, "api_key": self._key, "file_type": "json",
                                        "observation_start": start, "observation_end": end})
        try:
            with urllib.request.urlopen(f"{BASE_URL}?{query}", timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimited("FRED rate limit") from None
            raise TransientError(f"FRED HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise TransientError(f"FRED request failed: {type(e).__name__}") from None
        obs = payload.get("observations")
        if not isinstance(obs, list):
            raise SchemaDrift("FRED response has no observations list")
        return obs


def _fred_frame(obs: list[dict[str, str]]) -> pd.Series:
    rows: dict[str, float] = {}
    for o in obs:
        if set(o) < {"date", "value"}:
            raise SchemaDrift("FRED observation without date/value")
        try:
            v = float(o["value"])  # FRED writes "." for a missing day
        except ValueError:
            continue
        if v > 0:
            rows[str(o["date"])] = v
    return pd.Series(rows, dtype=float).sort_index()


def snapshot_bar_dates(fx: pd.DataFrame) -> pd.Series:
    """KRW per USD by Yahoo bar date. The bar is stamped one day after its date (end of the London day), see ``yfinance_adapter.fx_frame``."""
    # the stamp is local midnight of the NEXT London day, so convert to London time first: in summer it is 23:00 UTC of the bar's own date
    local = pd.to_datetime(fx.event_timestamp_utc, utc=True).dt.tz_convert(FX_BAR_TIMEZONE).dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)
    return pd.Series(fx.value.to_numpy(float), index=local.dt.strftime("%Y-%m-%d").to_numpy()).sort_index()


def fx_crosscheck(fx: pd.DataFrame, client: FredClient) -> dict[str, Any]:
    ours = snapshot_bar_dates(fx)
    first, last = str(ours.index[0]), str(ours.index[-1])
    theirs = _fred_frame(client.observations(SERIES_ID, first, last))
    inside = theirs[(theirs.index >= first) & (theirs.index <= last)]
    common = inside.index.intersection(ours.index)
    out: dict[str, Any] = {"series": SERIES_ID, "range": [first, last], "fred_observations": len(inside), "matched": len(common),
                           "criteria": {"min_matched_share": MIN_MATCHED_SHARE, "max_median_abs_rel_diff": MAX_MEDIAN_ABS_REL_DIFF,
                                        "max_p99_abs_rel_diff": MAX_P99_ABS_REL_DIFF, "max_abs_rel_diff": MAX_ABS_REL_DIFF}}
    failures: list[str] = []
    share = len(common) / len(inside) if len(inside) else 0.0
    # where the unmatched FRED dates fall, to tell a snapshot gap from a calendar or alignment cause (calendar facts, not prices)
    lost = inside.index.difference(ours.index)
    lost_dt = pd.to_datetime(list(lost)) if len(lost) else pd.DatetimeIndex([])
    out["snapshot_bars"] = len(ours)
    out["unmatched_by_year"] = {str(y): int(n) for y, n in pd.Series(lost_dt.year).value_counts().sort_index().items()}
    out["unmatched_by_weekday"] = {str(d): int(n) for d, n in pd.Series(lost_dt.day_name()).value_counts().items()}
    out["unmatched_first"] = [str(d) for d in lost[:30]]
    out["snapshot_only_bars"] = len(ours.index.difference(inside.index))
    out["matched_share"] = round(share, 4)
    if share < MIN_MATCHED_SHARE:
        failures.append(f"matched share {share:.3f} < {MIN_MATCHED_SHARE}")
    if len(common):
        rel = ((ours.loc[common].to_numpy() - inside.loc[common].to_numpy()) / inside.loc[common].to_numpy())
        a = np.abs(rel)
        out |= {"mean_rel_diff": round(float(rel.mean()), 5), "median_abs_rel_diff": round(float(np.median(a)), 5),
                "p99_abs_rel_diff": round(float(np.quantile(a, 0.99)), 5), "max_abs_rel_diff": round(float(a.max()), 5),
                "worst_dates": [str(d) for d in common[np.argsort(-a)[:5]]]}  # dates are calendar facts, not prices
        for name, limit in (("median_abs_rel_diff", MAX_MEDIAN_ABS_REL_DIFF), ("p99_abs_rel_diff", MAX_P99_ABS_REL_DIFF), ("max_abs_rel_diff", MAX_ABS_REL_DIFF)):
            if out[name] > limit:
                failures.append(f"{name} {out[name]} > {limit}")
    else:
        failures.append("no common dates")
    out["failures"] = failures
    out["passed"] = not failures
    return out


def run(snapshot: Path, client: FredClient | None = None) -> dict[str, Any]:
    fx = pd.read_parquet(snapshot / "fx_krw_per_usd.parquet")
    return fx_crosscheck(fx, client or RealFredClient())
