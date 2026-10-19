"""FRED DEXKOUS cross-check against hand-made responses (no network, no real data): matching, criteria, and that no key reaches a message."""
from __future__ import annotations

import urllib.error

import numpy as np
import pandas as pd
import pytest

from quant_pipeline import fred_crosscheck as F
from quant_pipeline.collector import ProviderError, RateLimited, SchemaDrift, TransientError

DAYS = pd.bdate_range("2021-01-04", "2021-12-31")


def snapshot_fx(values: np.ndarray) -> pd.DataFrame:
    """Yahoo bars are stamped one day after their date (end of the London day)."""
    ev = [(d + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z") for d in DAYS]
    return pd.DataFrame({"pair_id": "USDKRW", "event_timestamp_utc": ev, "available_at_utc": ev, "value": values,
                         "quote_convention": "KRW_per_USD", "source_id": "x"})


class FakeFred:
    def __init__(self, values, missing=()):
        self.values, self.missing = values, set(missing)

    def observations(self, series_id, start, end):
        return [{"date": d.strftime("%Y-%m-%d"), "value": "." if i in self.missing else f"{v:.4f}"} for i, (d, v) in enumerate(zip(DAYS, self.values, strict=True))]


BASE = 1100.0 + np.arange(len(DAYS)) * 0.1


def test_a_small_intraday_difference_passes_and_dates_align():
    ours = BASE * (1 + 0.001 * np.sin(np.arange(len(DAYS))))
    r = F.fx_crosscheck(snapshot_fx(ours), FakeFred(BASE))
    assert r["passed"], r["failures"]
    assert r["matched"] == len(DAYS) and r["max_abs_rel_diff"] < 0.002


def test_a_one_day_shift_is_visible_in_the_differences():
    steep = 1100.0 + np.arange(len(DAYS)) * 3.0
    r = F.fx_crosscheck(snapshot_fx(np.roll(steep, 1)), FakeFred(steep))
    assert r["median_abs_rel_diff"] > 0.0005  # a one-day misalignment of a trending series is not zero
    assert F.fx_crosscheck(snapshot_fx(steep), FakeFred(steep))["median_abs_rel_diff"] == 0.0


def test_a_level_error_fails_each_criterion_without_loosening():
    r = F.fx_crosscheck(snapshot_fx(BASE * 1.08), FakeFred(BASE))
    assert not r["passed"] and any("median" in f for f in r["failures"]) and any("max_abs" in f for f in r["failures"])


def test_missing_snapshot_dates_fail_the_matched_share():
    fx = snapshot_fx(BASE).iloc[::3].reset_index(drop=True)
    r = F.fx_crosscheck(fx, FakeFred(BASE))
    assert not r["passed"] and r["matched_share"] < F.MIN_MATCHED_SHARE


def test_fred_missing_marker_is_skipped_not_zero():
    r = F.fx_crosscheck(snapshot_fx(BASE), FakeFred(BASE, missing=range(40)))
    assert r["fred_observations"] == len(DAYS) - 40 and r["passed"]


def test_the_report_holds_no_series_values():
    r = F.fx_crosscheck(snapshot_fx(BASE), FakeFred(BASE))
    assert all(not isinstance(v, list) or k in {"range", "worst_dates", "failures", "unmatched_first"} for k, v in r.items())


def test_unmatched_dates_are_located_by_year_and_weekday():
    fx = snapshot_fx(BASE)
    fx = fx[~fx.event_timestamp_utc.str.startswith("2021-03-0")].reset_index(drop=True)  # drops the bars of 2021-03-01..09 minus one day
    r = F.fx_crosscheck(fx, FakeFred(BASE))
    assert r["unmatched_by_year"] == {"2021": sum(r["unmatched_by_weekday"].values())} and r["unmatched_first"][0].startswith("2021-03-0")
    assert r["snapshot_bars"] == len(fx) and r["snapshot_only_bars"] == 0


def test_a_malformed_response_is_schema_drift():
    class Bad:
        def observations(self, *a):
            return [{"date": "2021-01-04"}]

    with pytest.raises(SchemaDrift):
        F.fx_crosscheck(snapshot_fx(BASE), Bad())


def test_missing_key_refuses_and_http_errors_do_not_echo_the_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        F.RealFredClient()

    def boom(url, timeout):
        raise urllib.error.HTTPError(url, 429, "limit", {}, None)

    monkeypatch.setattr(F.urllib.request, "urlopen", boom)
    with pytest.raises(RateLimited) as e:
        F.RealFredClient("SECRETKEY123").observations("DEXKOUS", "2021-01-01", "2021-12-31")
    assert "SECRETKEY123" not in str(e.value)

    def down(url, timeout):
        raise urllib.error.URLError("SECRETKEY123 unreachable")

    monkeypatch.setattr(F.urllib.request, "urlopen", down)
    with pytest.raises(TransientError) as e2:
        F.RealFredClient("SECRETKEY123").observations("DEXKOUS", "2021-01-01", "2021-12-31")
    assert "SECRETKEY123" not in str(e2.value)
