"""The jump report lists the largest close-to-close moves per series inside a window (tools/price_jump_report.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("price_jump_report", ROOT / "tools" / "price_jump_report.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)


def prices(values, field="close"):
    days = pd.bdate_range("2025-05-01", periods=len(values))
    ev = [f"{d.date()}T00:00:00Z" for d in days]
    return pd.DataFrame({"instrument_id": "A", "event_timestamp_utc": ev, "field": field, "value": values})


def test_the_largest_move_is_found_and_dated_at_its_later_observation():
    df = prices([100.0, 101.0, 303.0, 304.0])
    r = J.largest_moves(df.assign(series="A|close"), "series", "value", "2025-05-01", "2025-05-30")
    top = r["A|close"]["largest"][0]
    assert top == ("2025-05-05", 2.0) and r["A|close"]["over_20pct"] == 1


def test_moves_outside_the_window_are_ignored():
    df = prices([100.0, 300.0, 301.0, 302.0])
    r = J.largest_moves(df.assign(series="A|close"), "series", "value", "2025-05-05", "2025-05-30")
    assert r["A|close"]["over_20pct"] == 0
