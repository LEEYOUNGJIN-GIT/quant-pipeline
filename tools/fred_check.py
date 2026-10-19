"""Compare a collected snapshot's USD/KRW series with FRED DEXKOUS (needs FRED_API_KEY); aggregate statistics only, exit 1 on failure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.collector import ProviderError
from quant_pipeline.fred_crosscheck import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    try:
        report = run(args.snapshot.resolve())
    except ProviderError as e:
        print(f"FRED cross-check could not run: {type(e).__name__}: {e}")
        return 2
    print(json.dumps(report, indent=1, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("fred cross-check:", "pass" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
