"""The overlap comparison of two snapshots reports differences without printing values (tools/snapshot_prefix_diff.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("snapshot_prefix_diff", ROOT / "tools" / "snapshot_prefix_diff.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def fx(days, values):
    ev = [f"{d}T00:00:00Z" for d in days]
    return pd.DataFrame({"pair_id": "USDKRW", "event_timestamp_utc": ev, "available_at_utc": ev, "value": values, "quote_convention": "x", "source_id": "y"})


DAYS = ["2024-12-26", "2024-12-27", "2024-12-30", "2025-01-02"]


def test_identical_overlap_is_identical_even_when_the_new_snapshot_is_longer():
    old = fx(DAYS[:3], [1.0, 2.0, 3.0])
    new = fx(DAYS, [1.0, 2.0, 3.0, 4.0])
    r = D.compare_table("fx_krw_per_usd", old, new, "2024-12-31")
    assert r["identical"] and r["common"] == 3 and r["only_in_new"] == 0  # the 2025 row lies beyond --through


def test_a_revised_value_is_counted_and_its_size_reported():
    r = D.compare_table("fx_krw_per_usd", fx(DAYS[:3], [1.0, 2.0, 3.0]), fx(DAYS[:3], [1.0, 2.02, 3.0]), "2024-12-31")
    assert not r["identical"] and r["numeric_cells_differing"] == 1 and abs(r["max_relative_difference"] - 0.01) < 1e-9


def test_missing_and_extra_rows_are_counted():
    r = D.compare_table("fx_krw_per_usd", fx(DAYS[:3], [1.0, 2.0, 3.0]), fx(DAYS[1:3], [2.0, 3.0]), "2024-12-31")
    assert r["only_in_old"] == 1 and not r["identical"]
