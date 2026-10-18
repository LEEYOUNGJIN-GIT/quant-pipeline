"""Verify a finished research/fixture run directory (integrity, lineage, labels, seal); optionally compare it byte-for-byte with a second run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.artifact_check import MAX_ABS_DIFFERENCE, check_run, compare_numeric, compare_runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="run directory (contains checkpoints/)")
    ap.add_argument("--compare", type=Path, default=None, help="a second run directory that must have byte-identical artifacts")
    ap.add_argument("--compare-numeric", action="store_true", help="with --compare: allow numeric-only differences (other hardware) but fail on any categorical one")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = check_run(args.run)
    if args.compare:
        cmp = compare_runs(args.run, args.compare)
        report["comparison"] = cmp
        if args.compare_numeric:
            num = compare_numeric(args.run, args.compare)
            report["numeric_comparison"] = num
            report["passed"] = report["passed"] and cmp["same_files"] and num["no_categorical_difference"] and num["within_numeric_bound"]
        else:
            report["passed"] = report["passed"] and cmp["identical"]
    for c in report["checks"]:
        print(("ok   " if c["ok"] else "FAIL ") + f"{c['name']}: {c['detail']}")
    if "comparison" in report:
        print(("ok   " if report["comparison"]["identical"] else "FAIL ") + f"byte comparison: {report['comparison']['n_different']} differing artifact(s)")
    if "numeric_comparison" in report:
        n = report["numeric_comparison"]
        print(f"     {n['n_differing_files']} differing artifacts ({n['n_json']} JSON, {n['n_other']} other): max relative difference {n['max_rel']:.3e}, "
              f"max absolute {n['max_abs']:.3e}; categorical differences in {len(n['categorical'])} file(s)")
        for k, r in n["worst_numeric"]:
            print(f"     numeric {r:.3e}  {k}")
        for field, count in n["categorical_fields"]:
            print(f"     CATEGORICAL x{count}  {field}")
        for field, count in n["derived_hash_differences"]:
            print(f"     derived hash x{count}  {field}")
        print(f"     numeric bound {MAX_ABS_DIFFERENCE:.0e} absolute: {'within' if n['within_numeric_bound'] else 'EXCEEDED'}")
        for k in n["other_files"]:
            print(f"     text {k}")
    print("artifacts:", "pass" if report["passed"] else "FAIL")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
