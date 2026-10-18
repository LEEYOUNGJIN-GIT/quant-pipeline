from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from quant_pipeline.calendars import Calendars
from quant_pipeline.config import load_config
from quant_pipeline.data import NoDataError, StaleDataError, load_initial_holdings, load_market_data
from quant_pipeline.event_time import Event, sort_events
from quant_pipeline.rates import RateCurve, accrue, act_days

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
T = pd.Timestamp


def test_accrual_one_day_matches_hand_calculation():
    assert accrue(Decimal("1000000000.00"), Decimal("0.035"), T("2024-01-02T21:15Z"), T("2024-01-03T21:15Z")) == Decimal("1000095890.41")


def test_same_day_interval_accrues_nothing_and_backwards_interval_fails():
    cash = Decimal("100.00")
    assert accrue(cash, Decimal("0.05"), T("2024-01-02T06:30Z"), T("2024-01-02T21:00Z")) == cash
    with pytest.raises(ValueError):
        act_days(T("2024-01-03T00:00Z"), T("2024-01-02T00:00Z"))


def test_negative_cash_cannot_accrue():
    with pytest.raises(ValueError):
        accrue(Decimal(-1), Decimal("0.03"), T("2024-01-02T00:00Z"), T("2024-01-03T00:00Z"))


def test_rate_is_not_usable_before_publication_and_stale_carry_halts():
    md = load_market_data(CFG, ROOT)
    curve = RateCurve.from_frame(md.rates, CFG.availability.max_rate_stale_business_days)
    with pytest.raises(NoDataError):
        curve.rate_in_force(T("2023-12-27T00:00Z"))
    assert curve.rate_in_force(T("2024-01-02T00:00Z")) == Decimal("0.035")  # available the instant it is published
    cals = Calendars.load(["XKRX", "XNYS"])
    short = RateCurve(curve.available[:1], curve.dates[:1], curve.annual[:1], 5)
    with pytest.raises(StaleDataError):
        short.rate_in_force(T("2024-03-01T00:00Z"), cals)


def test_negative_rates_are_rejected():
    df = pd.read_parquet(ROOT / CFG.data.rates).assign(annual_rate=-0.01)
    with pytest.raises(ValueError, match="financing spread"):
        RateCurve.from_frame(df, 5)


def test_fx_quote_older_than_limit_halts_and_missing_quote_fails():
    md = load_market_data(CFG, ROOT)
    assert md.fx_mark(T("2024-02-01T21:15Z")) > 1000
    with pytest.raises(NoDataError):
        md.fx_mark(T("2023-12-01T00:00Z"))
    with pytest.raises(StaleDataError):
        md.fx_mark(T("2024-05-30T00:00Z"))  # last quote is 2024-05-20: older than 72 hours


def test_liquidity_invalid_close_is_not_executable():
    md = load_market_data(CFG, ROOT)
    close = T("2024-03-14T20:00:00Z")  # NYSE_C flagged liquidity_invalid (EDT close)
    with pytest.raises(NoDataError, match="liquidity_invalid"):
        md.raw_close_at("NYSE_C", close)
    assert md.raw_close_at("NYSE_C", T("2024-03-13T20:00:00Z")) > 0


def test_raw_close_requires_an_exact_event_timestamp():
    md = load_market_data(CFG, ROOT)
    with pytest.raises(NoDataError):
        md.raw_close_at("KRX_A", T("2024-02-01T06:31:00Z"))


def test_duplicate_event_key_is_an_error_and_ids_are_sequential():
    a = Event(T("2024-01-01T00:00Z"), "valuation", "", "P", "X")
    with pytest.raises(ValueError, match="duplicate"):
        sort_events([a, Event(T("2024-01-01T00:00Z"), "valuation", "", "P", "X")])
    ids = [e.event_id for e in sort_events([a, Event(T("2024-01-02T00:00Z"), "valuation", "", "P", "Y")])]
    assert ids == ["E00001", "E00002"]


def test_initial_holdings_validation(tmp_path):
    cash, qty, as_of = load_initial_holdings(CFG, ROOT)
    assert cash == Decimal("1000000000.00") and set(qty) == {"KRX_A", "KRX_B", "NYSE_C"} and as_of.tzinfo is not None
    root = tmp_path
    (root / "data/fixtures/fixture_3asset").mkdir(parents=True)
    path = root / CFG.data.initial_holdings
    good = (ROOT / CFG.data.initial_holdings).read_text(encoding="utf-8")
    for bad, msg in [(good + "KRX_A,0,KRW,,2024-01-02T21:15:00Z\n", "duplicate"),
                     (good + "GHOST,0,KRW,,2024-01-02T21:15:00Z\n", "unknown"),
                     (good.replace("1000000000.00", "-5.00"), "non-negative")]:
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            load_initial_holdings(CFG, root)
