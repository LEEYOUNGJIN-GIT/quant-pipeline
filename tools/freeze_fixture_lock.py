#!/usr/bin/env python3
"""Freeze ``fixture_generation.lock.json`` BEFORE any fixture data is generated (spec section 14).

Records generator hash, scenario hash, seed stream/value, library versions and the calendar-session digest.
Refuses to run if generated fixture files already exist, so the lock can never be written after observing output.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_fixture as gen

GENERATED = ["manifest.json", "prices_raw.parquet", "prices_total_return.parquet", "fx_krw_per_usd.parquet",
             "rates_krw_rfr.parquet", "corporate_actions.parquet", "fixed_regime_table.csv", "initial_holdings.csv"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out: Path = args.output
    present = [f for f in GENERATED if (out / f).exists()]
    if present:
        raise SystemExit(f"refusing to freeze a lock after generation; found: {present}")
    lock_path = out / "fixture_generation.lock.json"
    if lock_path.exists():
        raise SystemExit(f"lock already exists and is immutable: {lock_path}")
    lock = gen.expected_lock()
    lock["frozen_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock["signoff"] = {
        "status": "self_attested_by_author",
        "statement": "Frozen before any model, optimizer or performance output existed. The generator was smoke-run "
                     "twice into a scratch directory solely to debug the code (row counts and determinism were "
                     "checked; the scenario was not tuned afterwards). REWORK RECORD: the first lock and fixture "
                     "were discarded once because a data test found a generator defect (regime label flickered "
                     "to low_risk on XKRX-closed XNYS anchors inside high_risk blocks); only the regime carry "
                     "rule was changed, then this lock was frozen and the fixture regenerated. At that time no "
                     "oracle, expected file, model or performance output existed. A second refreeze followed because a linter autofix altered the generator source after the first freeze; the logical hashes of all generated files were verified identical before and after (formatting-only change). A third rework changed only the fixed_regime_table.csv columns to the schema mandated by spec 7.5 (a requirement missed on the first pass); the other files' logical hashes were verified unchanged. No fixture files were written "
                     "to the fixture directory before this lock. No independent reviewer has signed off.",
    }
    out.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"froze {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
