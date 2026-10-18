"""Verify a finished research/fixture run directory (integrity, lineage, labels, seal); optionally compare it byte-for-byte with a second run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.artifact_check import check_run, compare_runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="run directory (contains checkpoints/)")
    ap.add_argument("--compare", type=Path, default=None, help="a second run directory that must have byte-identical artifacts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = check_run(args.run)
    if args.compare:
        cmp = compare_runs(args.run, args.compare)
        report["comparison"] = cmp
        report["passed"] = report["passed"] and cmp["identical"]
    for c in report["checks"]:
        print(("ok   " if c["ok"] else "FAIL ") + f"{c['name']}: {c['detail']}")
    if "comparison" in report:
        print(("ok   " if report["comparison"]["identical"] else "FAIL ") + f"byte comparison: {report['comparison']['n_different']} differing artifact(s)")
    print("artifacts:", "pass" if report["passed"] else "FAIL")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
