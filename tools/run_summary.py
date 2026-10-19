"""Aggregate summary of a finished research run directory, safe for a public log (no prices, weights or per-date values).

Prints the evidence class and snapshot, the run's geometry and the bootstrap statistics of the strategy and the benchmarks. These are pre-holdout results
of a single path through the data: descriptive, not a test of anything (see the report's limitations).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find(run: Path, name: str) -> Path | None:
    hits = sorted(run.rglob(name))
    return hits[0] if hits else None


def main(argv: list[str]) -> int:
    run = Path(argv[1])
    out: dict[str, object] = {"run": run.name}
    summary = find(run, "summary.json")
    coverage = find(run, "fixture_coverage.json")
    if summary is None:
        print("no summary.json: the run did not reach the bootstrap stage")
        return 1
    s = json.loads(summary.read_text(encoding="utf-8"))
    out |= {"evidence_class": s.get("evidence_class"), "snapshot": str(s.get("snapshot_id"))[:19], "bootstrap_status": s.get("bootstrap_status")}
    if coverage is not None:
        c = json.loads(coverage.read_text(encoding="utf-8"))
        out |= {k: c.get(k) for k in ("model_positions", "decision_rows", "outer_folds", "trials", "trial_space", "holdout")}
    print(json.dumps(out, indent=1, default=str))
    print("annualized Sharpe of returns in excess of the risk-free leg, pre-holdout; point estimate and percentile interval from a stationary bootstrap:")
    for name, st in sorted(s.get("statistics", {}).items()):
        pt, lo, hi = st.get("point_sharpe"), st.get("ci_low"), st.get("ci_high")
        shown = "undefined" if pt is None else f"{pt:7.3f}  [{lo:7.3f}, {hi:7.3f}]"
        print(f"  {name:40s} {shown}  replicates={st.get('replicates'):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
