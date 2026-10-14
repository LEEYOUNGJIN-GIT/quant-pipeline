"""Pairwise agent diagnostics (spec 8.2, check ``INV-AGENT-004``).

Per scope (calibration, outer-OOS, holdout): the valid simultaneous decision count; the Pearson correlation of each asset's absolute
target through time; the correlation of proposed risky budgets; and the per-decision cosine similarity of the risky-sleeve shares with
its median, 95th percentile and fraction above 0.99. Undefined cases are null with a reason, never zero. Diagnostic only: nothing here
changes a weight, and the spec gives no numeric threshold for "persistent identity" (OQ-018), so none is invented.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from .agents import AGENTS, CASH
from .agents.base import AgentOutput

COSINE_FLAG = 0.99


def _corr(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) < 3:
        return {"value": None, "reason": f"fewer than 3 simultaneous decisions ({len(a)})"}
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"value": None, "reason": "a constant series has no defined correlation"}
    return {"value": float(np.corrcoef(a, b)[0, 1]), "reason": None}


def _shares(out: AgentOutput, assets: tuple[str, ...]) -> np.ndarray | None:
    v = np.array([abs(out.absolute_target[a]) for a in assets])
    return None if v.sum() <= 0 else v / v.sum()


def pairwise(decisions: Sequence[Mapping[str, AgentOutput]], assets: tuple[str, ...]) -> dict[str, Any]:
    """``decisions``: one mapping agent name -> output per decision, containing only decisions where every agent produced a target."""
    valid = [d for d in decisions if all(a in d for a in AGENTS)]
    pairs: dict[str, Any] = {}
    for p, q in combinations(AGENTS, 2):
        per_asset = {a: _corr([d[p].absolute_target[a] for d in valid], [d[q].absolute_target[a] for d in valid]) for a in (*assets, CASH)}
        budget = _corr([d[p].proposed_risky_budget for d in valid], [d[q].proposed_risky_budget for d in valid])
        cos = []
        for d in valid:
            x, y = _shares(d[p], assets), _shares(d[q], assets)
            if x is not None and y is not None:
                cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
        c = np.array(cos)
        if len(c):
            cosine = {"count": len(c), "median": float(np.median(c)), "p95": float(np.percentile(c, 95)),
                      "fraction_above_0.99": float((c > COSINE_FLAG).mean()), "reason": None}
        else:
            cosine = {"count": 0, "median": None, "p95": None, "fraction_above_0.99": None,
                      "reason": "no decision where both agents hold a risky sleeve"}
        pairs[f"{p}|{q}"] = {"asset_target_correlation": per_asset, "risky_budget_correlation": budget, "risky_share_cosine": cosine}
    return {"valid_simultaneous_decisions": len(valid), "decisions_offered": len(decisions), "pairs": pairs}


def scope_report(engine: Any, scopes: Mapping[str, Sequence[int]], trial: Any) -> dict[str, Any]:
    """Run the four agents at each scope's decision positions with one candidate (the frozen/winning one)."""
    assets = engine.assets
    out: dict[str, Any] = {}
    for scope, positions in scopes.items():
        rows = []
        for pos in positions:
            fit = engine.fit(engine.refit_position(pos))
            target, _ = engine.target(pos, trial, fit)
            rows.append({o.agent_name: o for o in target.outputs})
        out[scope] = {"positions": [min(positions), max(positions)] if positions else None, **pairwise(rows, assets)}
    return {"schema": "agent-pairwise-diagnostics-1", "assets": list(assets), "agents": list(AGENTS),
            "identity_threshold": None, "threshold_note": "the specification gives no numeric threshold for persistent identity (OQ-018)",
            "scopes": out}


def max_flagged_pairs(report: Mapping[str, Any]) -> dict[str, list[str]]:
    """Pairs whose median cosine exceeds the 0.99 reporting line, per scope (a flag for review, not a verdict)."""
    return {s: [k for k, v in r["pairs"].items() if (v["risky_share_cosine"]["median"] or 0) > COSINE_FLAG] for s, r in report["scopes"].items()}
