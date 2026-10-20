"""FX freshness report: the limit is the pipeline's own rule and is not loosened; the report names the closes that break it (dates only)."""
from __future__ import annotations

import pandas as pd

from quant_pipeline.calendars import Calendars
from quant_pipeline.data_quality_report import MAX_FX_AGE_HOURS, fx_report

CALS = Calendars.load(["XKRX", "XNYS"])


def quotes(days) -> pd.DataFrame:
    """One quote per weekday stamped at the end of that London day, available 15 minutes later (as the live adapter does)."""
    ev = [(pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1)) for d in days]
    return pd.DataFrame({"pair_id": "USDKRW", "event_timestamp_utc": [e.strftime("%Y-%m-%dT%H:%M:%SZ") for e in ev],
                         "available_at_utc": [(e + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ") for e in ev],
                         "value": 1300.0, "quote_convention": "KRW_per_USD", "source_id": "x"})


FULL = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-03-01", "2023-04-28")]


def test_a_complete_weekday_series_passes_and_names_no_close():
    r = fx_report(quotes(FULL), CALS)
    assert not r["failures"] and r["closes_over_limit"] == 0 and r["closes_over_limit_first"] == []
    assert r["max_age_at_close_hours"] <= MAX_FX_AGE_HOURS


def test_a_missing_week_fails_and_the_report_names_the_closes_without_changing_the_limit():
    days = [d for d in FULL if not ("2023-03-27" <= d <= "2023-04-03")]  # a week without quotes
    r = fx_report(quotes(days), CALS)
    assert r["failures"] and r["closes_over_limit"] > 0 and set(r["closes_over_limit_by_market"]) == {"XKRX", "XNYS"}
    dates = [x.split()[1] for x in r["closes_over_limit_first"]]
    assert all("2023-03-27" <= d <= "2023-04-07" for d in dates), dates
    assert r["closes_over_limit"] == sum(r["closes_over_limit_by_market"].values())
    assert fx_report(quotes(days), CALS, max_age_hours=10_000)["failures"] == []  # only an explicit argument changes the limit
