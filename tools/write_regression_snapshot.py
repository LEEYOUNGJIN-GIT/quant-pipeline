#!/usr/bin/env python3
"""Explicit, reviewed regeneration of the fixture regression snapshot (never called by verification or production code).

Usage: python tools/write_regression_snapshot.py --projection PATH   (PATH from ``quant_pipeline.pipeline --projection-out``)
The lineage block is removed: lineage is compared against the freshly computed hashes of the verified tree, not stored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests" / "expected" / "fixture_3asset" / "regression"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection", required=True, type=Path)
    a = ap.parse_args()
    proj = json.loads(a.projection.read_text(encoding="utf-8"))
    proj.pop("lineage")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "projection.json").write_text(json.dumps(proj, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")
    (DEST / "projection_spec.json").write_text(json.dumps({
        "schema": "comparison-projection-1", "label": "reviewed_regression", "excluded_fields": [],
        "note": "production-dependent snapshot; never an independent oracle; every field of the projection is compared exactly",
        "lineage_policy": "compared with the hashes freshly computed for the verified tree"}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
