"""Deterministic (de)serialization of engine and evaluation results for checkpoints.

Decimals are stored as strings, timestamps as ISO-8601, NaN as ``null``. Loaded path results keep everything needed to score,
to carry state across a fold, and to compare ledgers; rebalance plans are kept as audit JSON (hashes and orders) and are not
re-materialised.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from .engine import DecisionRecord, FillRecord, PathCarry, PathResult
from .evaluation import InnerEvaluation, OuterCarry, OuterRun, PassOutcome
from .event_time import Event
from .replay import RunnerState
from .trials import Selection


def _clean(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return _clean(o.item())
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def dumps(o: Any) -> bytes:
    return (json.dumps(_clean(o), sort_keys=True, indent=1) + "\n").encode("utf-8")


def loads(b: bytes) -> Any:
    return json.loads(b.decode("utf-8"))


def _nan(x: float | None) -> float:
    return float("nan") if x is None else x


# ------------------------------------------------------------------------------------------------------ carry / events
def carry_to_obj(c: PathCarry) -> dict[str, Any]:
    r = c.runner
    return {"runner": {"cash": str(r.cash), "holdings": r.holdings, "last_ts": r.last_ts.isoformat(),
                       "cumulative_costs": str(r.cumulative_costs), "entitlement": r.entitlement},
            "pending": [{"ts": e.ts.isoformat(), "kind": e.kind, "exchange": e.exchange, "asset_id": e.asset_id,
                         "order_id": e.order_id, "payload": e.payload, "event_id": e.event_id} for e in c.pending],
            "drained_ts": c.drained_ts.isoformat(), "last_weights": None if c.last_weights is None else c.last_weights.tolist(),
            "policy_state": c.policy_state}


def carry_from_obj(o: dict[str, Any]) -> PathCarry:
    r = o["runner"]
    state = RunnerState(Decimal(r["cash"]), {k: int(v) for k, v in r["holdings"].items()}, pd.Timestamp(r["last_ts"]),
                        Decimal(r["cumulative_costs"]), {k: int(v) for k, v in r["entitlement"].items()})
    pending = [Event(pd.Timestamp(e["ts"]), e["kind"], e["exchange"], e["asset_id"], e["order_id"], e["payload"], e["event_id"])
               for e in o["pending"]]
    lw = None if o["last_weights"] is None else np.array(o["last_weights"], dtype=float)
    return PathCarry(state, pending, pd.Timestamp(o["drained_ts"]), lw, o["policy_state"])


# ------------------------------------------------------------------------------------------------------ path results
def path_to_artifacts(r: PathResult, prefix: str = "") -> dict[str, bytes]:
    plans = [{"decision_id": p.decision_id, "plan_hash": p.plan_content_hash, "fold_id": p.fold_id, "refit_id": p.refit_id,
              "orders": [o.__dict__ for o in p.orders], "summary": p.summary} for p in r.plans]
    meta = {"label": r.label, "costs": str(r.costs), "optimizer_attempts": r.optimizer_attempts}
    recon = [{"order_id": x["order_id"], "cost": str(x["cost"]), "nav_change_plus_cost": str(x["nav_change_plus_cost"]),
              "within_tolerance": bool(x["within_tolerance"])} for x in r.reconciliation]
    ledger = r.ledger_rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {f"{prefix}decisions.json": dumps([asdict(d) for d in r.decisions]), f"{prefix}fills.json": dumps([asdict(f) for f in r.fills]),
            f"{prefix}nav.json": dumps({k: str(v) for k, v in r.nav.items()}), f"{prefix}ledger_rows.csv": ledger,
            f"{prefix}plans.json": dumps(plans), f"{prefix}carry.json": dumps(carry_to_obj(r.carry)),
            f"{prefix}meta.json": dumps(meta), f"{prefix}reconciliation.json": dumps(recon)}


def path_from_artifacts(a: dict[str, bytes], prefix: str = "") -> PathResult:
    decisions = []
    for d in loads(a[f"{prefix}decisions.json"]):
        d["objective_terms"] = None if d["objective_terms"] is None else tuple(d["objective_terms"])
        decisions.append(DecisionRecord(**d))
    fills = [FillRecord(**f) for f in loads(a[f"{prefix}fills.json"])]
    nav = {k: Decimal(v) for k, v in loads(a[f"{prefix}nav.json"]).items()}
    rows = pd.read_csv(io.BytesIO(a[f"{prefix}ledger_rows.csv"]), dtype=str, keep_default_na=False)
    meta = loads(a[f"{prefix}meta.json"])
    recon = [{"order_id": x["order_id"], "cost": Decimal(x["cost"]), "nav_change_plus_cost": Decimal(x["nav_change_plus_cost"]),
              "within_tolerance": x["within_tolerance"]} for x in loads(a[f"{prefix}reconciliation.json"])]
    return PathResult(meta["label"], decisions, fills, nav, rows, carry_from_obj(loads(a[f"{prefix}carry.json"])), recon, [],
                      meta["optimizer_attempts"], Decimal(meta["costs"]))


# ------------------------------------------------------------------------------------------------------ evaluation
def outcome_to_obj(o: PassOutcome) -> dict[str, Any]:
    s = o.selection
    return {"name": o.name, "train_end": o.train_end, "evaluations": [asdict(e) for e in o.evaluations],
            "selection": {"winner": s.winner, "ranking": list(s.ranking), "tie_path": list(s.tie_path), "margin": s.margin,
                          "mean_scores": s.mean_scores, "stable": s.stable},
            "winner_hash": o.winner_hash, "winner_parameters": o.winner_parameters,
            "loo": list(o.leave_one_fold_out_winners), "unstable": o.selection_unstable}


def outcome_from_obj(o: dict[str, Any]) -> PassOutcome:
    evals = []
    for e in o["evaluations"]:
        e["score"], e["turnover"] = _nan(e["score"]), _nan(e["turnover"])
        evals.append(InnerEvaluation(**e))
    s = o["selection"]
    sel = Selection(s["winner"], tuple(s["ranking"]), tuple(s["tie_path"]), _nan(s["margin"]), s["mean_scores"], s["stable"])
    return PassOutcome(o["name"], o["train_end"], tuple(evals), sel, o["winner_hash"], o["winner_parameters"], tuple(o["loo"]), o["unstable"])


def outer_to_artifacts(run: OuterRun, carry: OuterCarry) -> dict[str, bytes]:
    out = path_to_artifacts(run.result, "strategy.")
    for b, r in run.benchmarks.items():
        out.update(path_to_artifacts(r, f"bench.{b}."))
    out["outer_meta.json"] = dumps({"fold_id": run.fold_id, "trial_hash": run.trial_hash, "multiplier": run.multiplier,
                                    "benchmarks": sorted(run.benchmarks)})
    return out


def outer_from_artifacts(a: dict[str, bytes]) -> tuple[OuterRun, OuterCarry]:
    meta = loads(a["outer_meta.json"])
    run = OuterRun(meta["fold_id"], meta["trial_hash"], meta["multiplier"], path_from_artifacts(a, "strategy."))
    for b in meta["benchmarks"]:
        run.benchmarks[b] = path_from_artifacts(a, f"bench.{b}.")
    carry = OuterCarry(run.result.carry, {b: r.carry for b, r in run.benchmarks.items()}, {})
    return run, carry
