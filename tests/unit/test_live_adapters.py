"""Yahoo and ECOS adapters against hand-made responses (no network, no real data): the mapping must reproduce the project's conventions."""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline import ecos_adapter as E
from quant_pipeline import yfinance_adapter as Y
from quant_pipeline.collector import (
    ACTION_COLUMNS,
    FX_COLUMNS,
    PRICE_COLUMNS,
    RATE_COLUMNS,
    ProviderError,
    RateLimited,
    SchemaDrift,
    SymbolNotFound,
)
from support.fake_live import CALS, FakeEcos, FakeYahoo, yahoo_frame

START, END = "2021-01-04", "2022-06-30"


def prices(symbol="005930.KS", exchange="XKRX", **kw):
    h = yahoo_frame(symbol, exchange, START, END, **kw)
    cur = "KRW" if exchange == "XKRX" else "USD"
    return h, *Y.price_frames(symbol, h, exchange=exchange, currency=cur, cals=CALS, price_delay="PT15M")


def test_the_split_adjustment_is_undone_so_raw_prices_jump_on_the_split_day_and_total_return_does_not():
    h, raw, tr, diag = prices(split_on="2021-09-01", split_ratio=2.0, dividend_yield=0.0)
    day = raw.index[raw.exchange_session == "2021-09-01"][0]
    jump = raw.value.iloc[day] / raw.value.iloc[day - 1]
    assert 0.40 < jump < 0.60  # as traded: the price halves
    adj = h.frame.set_axis(h.frame.index.tz_localize(None).strftime("%Y-%m-%d")).loc[raw.exchange_session, "Close"].to_numpy(float)
    assert np.allclose(raw.value.to_numpy()[:day], adj[:day] * 2.0, rtol=1e-4)  # before the split the as-traded price is twice Yahoo's adjusted one
    assert np.allclose(raw.value.to_numpy()[day:], adj[day:], rtol=1e-4)  # from the split on the two agree
    r = tr.value.pct_change().abs()
    assert r.iloc[day] < 0.08  # the index only moves with the market there
    assert diag["splits"] == 1 and diag["dividends"] == 0


def test_total_return_follows_the_project_convention_dividends_reinvested_at_the_ex_date_close():
    h, raw, tr, _ = prices(exchange="XNYS", symbol="SPY", dividend_yield=0.01)
    by_day = h.frame.set_axis(h.frame.index.tz_localize(None).strftime("%Y-%m-%d"))
    close, div = by_day.loc[raw.exchange_session, "Close"].to_numpy(float), by_day.loc[raw.exchange_session, "Dividends"].to_numpy(float)
    i = int(np.flatnonzero(div > 0)[0])
    growth = tr.value.iloc[i] / tr.value.iloc[i - 1]
    assert growth == pytest.approx((close[i] + div[i]) / close[i - 1], rel=1e-9)  # not Yahoo's adj-close factor, which reinvests at the previous close
    assert tr.value.iloc[0] == pytest.approx(100.0)


def test_columns_fields_timestamps_and_flags_match_the_snapshot_contract():
    _, raw, tr, diag = prices()
    assert list(raw.columns) == PRICE_COLUMNS and list(tr.columns) == PRICE_COLUMNS
    assert set(raw.field) == {"close"} and set(tr.field) == {"total_return_index"}
    first = raw.iloc[0]
    close = CALS.close("XKRX", pd.Timestamp(first.exchange_session))
    assert first.event_timestamp_utc == close.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert first.available_at_utc == (close + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert (raw.quality_flag == "liquidity_invalid").sum() == diag["zero_volume_bars"] == 1 and set(tr.quality_flag) == {"ok"}
    assert diag["non_session_rows"] == 1  # the bar on a day that is not an official session is dropped and counted, not kept
    assert set(raw.exchange_session) <= {d.strftime("%Y-%m-%d") for d in CALS.sessions("XKRX", START, END)}
    assert (raw.value > 0).all() and raw.unit.iloc[0] == "KRW/share"


def test_actions_use_as_traded_units_the_ex_session_close_and_rational_ratios():
    h = yahoo_frame("005930.KS", "XKRX", START, END, split_on="2021-09-01", split_ratio=2.0, dividend_every=100, dividend_yield=0.01)
    acts = Y.action_frame("005930.KS", h, exchange="XKRX", currency="KRW", cals=CALS)
    assert list(acts.columns) == ACTION_COLUMNS
    s = acts[acts.action_type == "split"].iloc[0]
    assert (s.ratio_num, s.ratio_den) == (2.0, 1.0) and s.ex_session_date == "2021-09-01"
    prev = CALS.close("XKRX", pd.Timestamp("2021-08-31")).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert s.available_at_utc == prev and s.declared_at_utc == prev  # known at the previous session close, strictly before the ex timestamp
    assert s.ex_timestamp_utc == CALS.close("XKRX", pd.Timestamp("2021-09-01")).strftime("%Y-%m-%dT%H:%M:%SZ") and s.available_at_utc < s.ex_timestamp_utc
    divs = acts[acts.action_type == "cash_dividend"]
    first = divs.iloc[0]  # before the split: Yahoo's adjusted dividend times the split ratio is the cash actually paid per share
    adj = float(h.frame.loc[h.frame.index.tz_localize(None) == pd.Timestamp(first.ex_session_date), "Dividends"].iloc[0])
    assert first.cash_amount == pytest.approx(adj * 2.0, abs=0.02) and first.cash_currency == "KRW"


def test_a_reverse_split_becomes_a_ratio_below_one():
    h = yahoo_frame("TLT", "XNYS", START, END, split_on="2021-09-01", split_ratio=0.1, dividend_yield=0.0)
    s = Y.action_frame("TLT", h, exchange="XNYS", currency="USD", cals=CALS).query("action_type == 'split'").iloc[0]
    assert (s.ratio_num, s.ratio_den) == (1.0, 10.0)


def test_schema_drift_and_unusable_responses_fail_explicitly():
    h = yahoo_frame("SPY", "XNYS", START, END)
    with pytest.raises(SchemaDrift, match="lacks columns"):
        Y.price_frames("SPY", Y.YahooHistory(h.frame.drop(columns=["Dividends"]), "USD", "America/New_York"), exchange="XNYS", currency="USD", cals=CALS, price_delay="PT15M")
    with pytest.raises(SchemaDrift, match="quotes in"):
        Y.price_frames("SPY", Y.YahooHistory(h.frame, "EUR", "America/New_York"), exchange="XNYS", currency="USD", cals=CALS, price_delay="PT15M")
    with pytest.raises(SymbolNotFound):
        Y.price_frames("SPY", Y.YahooHistory(h.frame.iloc[0:0], "USD", None), exchange="XNYS", currency="USD", cals=CALS, price_delay="PT15M")
    bad = h.frame.copy()
    bad.loc[bad.index[10], "Stock Splits"] = 1.2345678
    with pytest.raises(SchemaDrift, match="rational"):
        Y.action_frame("SPY", Y.YahooHistory(bad, "USD", None), exchange="XNYS", currency="USD", cals=CALS)


def test_fx_quotes_are_stamped_at_the_end_of_the_london_day_and_a_timezone_change_is_refused():
    fake = FakeYahoo(START, END)
    h = fake.history("KRW=X", START, END)
    fx = Y.fx_frame("USDKRW", h, fx_delay="PT15M")
    assert list(fx.columns) == FX_COLUMNS and set(fx.pair_id) == {"USDKRW"} and set(fx.quote_convention) == {"KRW_per_USD"}
    assert fx.event_timestamp_utc.iloc[0] == "2021-01-05T00:00:00Z" and fx.available_at_utc.iloc[0] == "2021-01-05T00:15:00Z"  # winter: London == UTC
    assert fx.event_timestamp_utc.iloc[-1] > fx.event_timestamp_utc.iloc[0] and fx.value.gt(0).all()
    clipped = Y.fx_frame("USDKRW", h, fx_delay="PT15M", end="2021-03-02")  # winter: the 2 March bar closes on 3 March, outside the range
    assert clipped.event_timestamp_utc.iloc[-1] == "2021-03-02T00:00:00Z" and clipped.event_timestamp_utc.str[:10].max() <= "2021-03-02"
    with pytest.raises(SchemaDrift, match="timezone"):
        Y.fx_frame("USDKRW", Y.YahooHistory(h.frame, "KRW", "Asia/Seoul"), fx_delay="PT15M")


def test_ecos_rates_are_decimal_annual_simple_and_available_one_xkrx_session_later():
    rows = FakeEcos(3.5).series("817Y002", "010101000", "20210104", "20210115")
    frame, dropped = E.rates_frame(rows, cals=CALS, publication_delay_business_days=1)
    assert list(frame.columns) == RATE_COLUMNS and set(frame.annual_rate) == {0.035} and set(frame.quote_basis) == {"annual_simple"}
    assert set(frame.day_count) == {"ACT/365F"}
    sessions = [d.strftime("%Y-%m-%d") for d in CALS.sessions("XKRX", "2021-01-04", "2021-01-20")]
    for _, r in frame.iterrows():
        nxt = sessions[sessions.index(r.rate_date) + 1]
        assert r.available_at_utc == f"{nxt}T00:00:00Z" and r.event_timestamp_utc == f"{r.rate_date}T00:00:00Z"
    assert dropped["non_session_rows"] == 0 and dropped["unparseable_rows"] == 0
    wk = FakeEcos().series("a", "b", "20210101", "20210301")
    _, dropped = E.rates_frame(wk, cals=CALS, publication_delay_business_days=1)
    assert dropped["non_session_rows"] >= 1  # a weekday that is an XKRX holiday is dropped and counted


def test_ecos_errors_never_leak_the_key_and_map_to_provider_error_kinds(monkeypatch):
    client = E.RealEcosClient("SECRETKEY123")

    def boom(url, timeout=None):
        raise urllib.error.URLError(f"cannot reach {url}")

    monkeypatch.setattr(E.urllib.request, "urlopen", boom)
    with pytest.raises(ProviderError) as e:
        client.series("817Y002", "010101000", "20210101", "20210201")
    assert "SECRETKEY123" not in str(e.value) and "SECRETKEY123" not in repr(e.value.__cause__)
    with pytest.raises(SymbolNotFound):
        E.RealEcosClient._raise_for({"RESULT": {"CODE": "INFO-200", "MESSAGE": "no data"}})
    with pytest.raises(RateLimited):
        E.RealEcosClient._raise_for({"RESULT": {"CODE": "ERROR-602", "MESSAGE": "x"}})
    with pytest.raises(ProviderError) as e2:
        E.RealEcosClient._raise_for({"RESULT": {"CODE": "ERROR-100", "MESSAGE": "bad key SECRETKEY123"}})
    assert "SECRETKEY123" not in str(e2.value)
    with pytest.raises(ProviderError, match="not set"):
        monkeypatch.delenv("ECOS_API_KEY", raising=False)
        E.RealEcosClient()


def test_the_provider_terms_printed_in_the_research_report_pass_the_terminology_scan():
    """The first real-data run reached the pre-holdout report and stopped there: the Yahoo terms said 'no accuracy guarantee', and the scan forbids the
    word 'guarantee' (it cannot tell a disclaimer from a claim). Every provider's terms are printed in report section 10, so they must scan clean."""
    from quant_pipeline.ecos_adapter import EcosProvider
    from quant_pipeline.reporting import scan_terminology
    from quant_pipeline.yfinance_adapter import YahooProvider

    for terms in (YahooProvider.license, EcosProvider.license, f"{YahooProvider.license}; {EcosProvider.license}"):
        assert scan_terminology(terms)["forbidden"] == [], terms
    assert scan_terminology("no service-level or accuracy guarantee")["forbidden"] == ["proof / guarantee claim"]  # the old wording is what failed
