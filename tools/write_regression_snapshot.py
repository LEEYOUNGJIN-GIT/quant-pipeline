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
SPEC = json.loads('{"excluded_fields": [], "label": "reviewed_regression", "lineage_policy": "compared with the hashes freshly computed for the verified tree", "note": "production-dependent snapshot; never an independent oracle; every field is compared exactly except the two declared numeric tolerances below", "schema": "comparison-projection-1", "tolerances": {"plan_hashes": "count_only", "rationale": "the same code and lock gave different registry scores (max abs 1.7e-2, no eligibility flips) and therefore different plan hashes on different GitHub-hosted runners; fills, NAV, ledger hashes, costs, benchmarks, winners and eligibility are identical across all observed variants and stay exact", "registry_score_abs": 0.05}}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection", required=True, type=Path)
    a = ap.parse_args()
    proj = json.loads(a.projection.read_text(encoding="utf-8"))
    proj.pop("lineage")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "projection.json").write_text(json.dumps(proj, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")
    (DEST / "projection_spec.json").write_text(json.dumps(SPEC, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
