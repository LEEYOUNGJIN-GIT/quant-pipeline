"""Print and write the data-quality report of a collected research snapshot; exit 1 when a criterion fails (aggregate statistics only)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.config import load_yaml_strict
from quant_pipeline.data_quality_report import snapshot_report, write_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--dividend-payers", default="", help="comma-separated asset ids expected to pay dividends regularly")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    universe = load_yaml_strict(args.universe.read_text(encoding="utf-8"))["assets"]
    payers = frozenset(x for x in args.dividend_payers.split(",") if x)
    report = snapshot_report(args.snapshot.resolve(), universe, payers)
    for i in report["instruments"]:
        print(f"{i['symbol']:>10} {i['exchange']} bars={i['bars']} cov={i['session_coverage']:.4f} gap={i['max_consecutive_missing_sessions']} "
              f"div/yr={i['dividends_per_year']} splits={i['splits']} maxmove={i['max_abs_daily_return_excluding_splits']}")
    print("fx:", {k: v for k, v in report["fx"].items() if k != "failures"})
    print("rates:", {k: v for k, v in report["rates"].items() if k != "failures"})
    for f in report["failures"]:
        print("FAIL", f)
    print("data quality:", "pass" if report["passed"] else "FAIL")
    if args.out:
        write_report(report, args.out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
