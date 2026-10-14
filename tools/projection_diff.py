#!/usr/bin/env python3
"""Explain how a run's projection differs from the checked-in regression snapshot (diagnostic only; it decides nothing).

Usage: python tools/projection_diff.py PROJECTION.json [SNAPSHOT.json]
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "expected" / "fixture_3asset" / "regression" / "projection.json"


def main(argv: list[str]) -> int:
    actual = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    expected = json.loads(Path(argv[2] if len(argv) > 2 else SNAPSHOT).read_text(encoding="utf-8"))
    print("top-level equality:", {k: actual.get(k) == v for k, v in expected.items()})
    ra, re_ = actual["registry"], expected["registry"]
    diff = [k for k in re_ if ra.get(k) != re_[k]]
    sc = [abs(ra[k][0] - re_[k][0]) for k in diff if ra[k][0] is not None and re_[k][0] is not None]
    print(f"registry: {len(diff)} of {len(re_)} entries differ; max abs score diff {max(sc, default=0.0):.3e}; "
          f"eligibility flips {sum(1 for k in diff if ra[k][1] != re_[k][1])}")
    for key, e in expected["paths"].items():
        a = actual["paths"].get(key)
        if a == e:
            continue
        flags = {f: a[f] == e[f] for f in e}
        plans = sum(1 for x, y in zip(a["plans"], e["plans"], strict=False) if x != y)
        navs = [abs(Decimal(a["nav"][d]) - Decimal(e["nav"][d])) / Decimal(e["nav"][d]) for d in e["nav"] if d in a["nav"]]
        print(f"path {key}: equal fields {flags}; plans differing {plans}/{len(e['plans'])}; "
              f"max relative NAV diff {max(navs, default=Decimal(0)):.3e}; fills {len(a['fills'])} vs {len(e['fills'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
