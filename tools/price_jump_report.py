"""Largest one-day moves per series of a research snapshot (aggregate statistics only; prices themselves are never printed).

A forward test showed implausible single-interval returns. This separates a data problem from an accounting problem: it lists, per instrument, the
largest absolute close-to-close changes of the raw and the total-return series with their dates, the FX series' largest changes, and the corporate
actions on or after ``--since``. A control window can be given with ``--control-through``; dates after the sealed holdout's start must not be used there.
It judges nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOP = 3


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["event_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")


def largest_moves(df: pd.DataFrame, group: str, value: str, lo: str, hi: str, top: int = TOP) -> dict[str, Any]:
    """Per group: the ``top`` largest |close-to-close change| inside [lo, hi] (the change is dated at its later observation)."""
    out: dict[str, Any] = {}
    work = df.assign(_d=_dates(df)).sort_values([group, "_d"])
    for key, g in work.groupby(group):
        v = g[value].to_numpy(float)
        d = g["_d"].to_numpy()
        if len(v) < 2:
            out[str(key)] = {"observations": len(v), "largest": []}
            continue
        ch = v[1:] / v[:-1] - 1.0
        dd = d[1:]
        keep = (dd >= lo) & (dd <= hi) & np.isfinite(ch)
        ch, dd = ch[keep], dd[keep]
        order = np.argsort(-np.abs(ch))[:top]
        out[str(key)] = {"observations": int(keep.sum()), "over_20pct": int((np.abs(ch) > 0.2).sum()),
                         "largest": [(str(dd[i]), round(float(ch[i]), 4)) for i in order]}
    return out


def report(snapshot: Path, lo: str, hi: str, since: str) -> dict[str, Any]:
    raw = pd.read_parquet(snapshot / "prices_raw.parquet")
    tr = pd.read_parquet(snapshot / "prices_total_return.parquet")
    fx = pd.read_parquet(snapshot / "fx_krw_per_usd.parquet")
    ca = pd.read_parquet(snapshot / "corporate_actions.parquet")

    def keyed(df: pd.DataFrame) -> pd.DataFrame:  # one series per instrument and field
        return df.assign(series=df["instrument_id"].astype(str) + "|" + df["field"].astype(str))

    out: dict[str, Any] = {"window": [lo, hi],
                           "raw_close": largest_moves(keyed(raw), "series", "value", lo, hi),
                           "total_return": largest_moves(keyed(tr), "series", "value", lo, hi),
                           "fx": largest_moves(fx, "pair_id", "value", lo, hi)}
    after = ca[pd.to_datetime(ca["ex_timestamp_utc"], utc=True) >= pd.Timestamp(since, tz="UTC")]
    out["corporate_actions_since"] = {"since": since, "count": len(after),
                                      "by_type": {str(k): int(v) for k, v in after.groupby("action_type").size().items()},
                                      "splits": [(str(r.instrument_id), str(r.ex_session_date), f"{r.ratio_num}/{r.ratio_den}")
                                                 for r in after[after["action_type"] == "split"].itertuples()]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--from", dest="lo", default="2025-04-23")
    ap.add_argument("--to", dest="hi", default="2026-09-30")
    ap.add_argument("--control-from", default="2015-01-02")
    ap.add_argument("--control-through", default="2023-08-31", help="must end before the sealed holdout starts")
    ap.add_argument("--since", default="2025-01-02")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = {"forward_window": report(args.snapshot, args.lo, args.hi, args.since),
              "control_window": report(args.snapshot, args.control_from, args.control_through, args.control_from)}
    text = json.dumps(result, indent=1, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
