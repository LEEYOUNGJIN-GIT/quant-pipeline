"""Complete-path coverage matrix (spec 14, check ``INV-FIX-001``): what the fixture run actually exercised, counted from the run's own
results rather than asserted. A row with a zero count is reported as uncovered; nothing is filled in."""
from __future__ import annotations

from typing import Any


def coverage_matrix(*, manifest: Any, outcomes: list[Any], outer: dict[float, list[Any]], summary: dict[str, Any], refit_positions: tuple[int, ...],
                    refits_used: tuple[int, ...], checkpoint_stages: dict[str, int], generation_lock: dict[str, str]) -> dict[str, Any]:
    inner_evals = [e for o in outcomes for e in o.evaluations]
    runs1 = outer[1.0]
    rows: dict[str, dict[str, Any]] = {
        "inner_validation": {"passes": len(manifest.passes), "blocks": sum(len(p.inner) for p in manifest.passes),
                             "trial_fold_evaluations": len(inner_evals), "eligible": sum(e.eligible for e in inner_evals),
                             "realized_fills": sum(e.realized_fills for e in inner_evals)},
        "outer_oos": {"folds": len(runs1), "decisions": sum(len(r.result.decisions) for r in runs1),
                      "fills": sum(len(r.result.fills) for r in runs1), "scored_intervals": summary["oos_intervals"]["strategy"]},
        "holdout": {"geometry_decisions": manifest.holdout.n_decisions, "geometry_intervals": manifest.holdout.n_valuation_intervals,
                    "opened_in_this_matrix": False},
        "refit": {"scheduled_positions": len(refit_positions), "cores_used_by_decisions": len(refits_used)},
        "cost_scenarios": {f"{m:g}x": {"decisions": sum(len(r.result.decisions) for r in runs), "realized_costs": str(sum(r.result.costs for r in runs))}
                           for m, runs in sorted(outer.items())},
        "benchmarks": {b: summary["oos_intervals"][b] for b in runs1[0].benchmarks},
        "bootstrap": {"shape": summary["bootstrap_shape"], "index_hash": summary["bootstrap_index_hash"], "status": summary["bootstrap_status"]},
        "checkpoint_stages": dict(sorted(checkpoint_stages.items())),
    }
    uncovered = [k for k, v in rows.items() if k not in ("holdout",) and _is_empty(v)]
    return {"schema": "fixture-coverage-1", "generation_lock": generation_lock, "rows": rows, "uncovered": uncovered,
            "note": "the holdout row is geometry only; its execution is recorded in the audited opening"}


def _is_empty(v: Any) -> bool:
    if isinstance(v, dict):
        return not v or all(_is_empty(x) for x in v.values())
    return v in (0, [], None, "")
