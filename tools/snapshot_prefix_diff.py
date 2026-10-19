"""Compare the overlapping dates of two research snapshots (aggregate statistics only; never prints a price).

A forward test re-collects the earlier years together with the new ones. Yahoo may have revised adjusted prices or dividends in between, so the new
snapshot's early rows are not guaranteed to equal the ones the frozen selection saw. This reports how far they differ, per table, over the dates both
snapshots cover (up to ``--through``). It judges nothing: the numbers go into the forward-test record next to its result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.collector import SCHEMAS

TIME_COLUMN = {"prices_raw": "event_timestamp_utc", "prices_total_return": "event_timestamp_utc", "corporate_actions": "ex_timestamp_utc",
               "fx_krw_per_usd": "event_timestamp_utc", "rates_krw_rfr": "event_timestamp_utc"}


def compare_table(stem: str, old: pd.DataFrame, new: pd.DataFrame, through: str) -> dict[str, object]:
    key = SCHEMAS[stem][1]
    tc = TIME_COLUMN[stem]
    cut = pd.Timestamp(through, tz="UTC") + pd.Timedelta(days=1)

    def upto(df: pd.DataFrame) -> pd.DataFrame:
        t = pd.to_datetime(df[tc], utc=True)
        return df[t < cut].copy()

    a, b = upto(old), upto(new)
    ka = a.set_index(key).sort_index()
    kb = b.set_index(key).sort_index()
    only_old = ka.index.difference(kb.index)
    only_new = kb.index.difference(ka.index)
    both = ka.index.intersection(kb.index)
    out: dict[str, object] = {"table": stem, "rows_old": len(a), "rows_new": len(b), "only_in_old": len(only_old), "only_in_new": len(only_new),
                              "common": len(both)}
    numeric = [c for c in ka.columns if pd.api.types.is_numeric_dtype(ka[c]) and not pd.api.types.is_bool_dtype(ka[c]) and c in kb.columns]
    worst, differing = 0.0, 0
    for c in numeric:
        x, y = ka.loc[both, c].to_numpy(float), kb.loc[both, c].to_numpy(float)
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        differing += int((~same).sum())
        denom = np.where(np.abs(x) > 0, np.abs(x), 1.0)
        rel = np.where(same, 0.0, np.abs(x - y) / denom)
        worst = max(worst, float(np.nanmax(rel)) if rel.size else 0.0)
    out |= {"numeric_cells_differing": differing, "max_relative_difference": worst}
    out["identical"] = not (len(only_old) or len(only_new) or differing)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--through", default="2024-12-31")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = [compare_table(stem, pd.read_parquet(args.old / f"{stem}.parquet"), pd.read_parquet(args.new / f"{stem}.parquet"), args.through)
              for stem in SCHEMAS]
    text = json.dumps({"through": args.through, "tables": report, "all_identical": all(r["identical"] for r in report)}, indent=1, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
