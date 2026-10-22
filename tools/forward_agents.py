"""Agent-level view of the forward path's decisions (OQ-062): why the risky sleeve was built the way it was.

Nothing is selected, tuned or tested against a rule, and the numbers are not an evidence statement. The same frozen selection, snapshot and interval as
tools/forward_test.py are replayed; for each decision the aggregation step is read back from the engine's target cache (the engine computed it, nothing is
recomputed here) and summarised as shares and ratios only:

* each agent's normalised score (its view of the risky assets), proposed risky budget and confidence, by asset group;
* the regime gates (raw and normalised) that weight the agents, and the regime probabilities;
* the gate-weighted mixture ``q_bar`` of the agents' scores, the proposed budget, the risk-pressure envelope and which of
  ``min(b_proposed, b_defensive, 1 - cash floor)`` set the final budget (``b_final``);
* the target ``a_bar`` (before the optimiser) against the optimiser's weights and the holdings before the trade, by asset group;
* the optimiser's objective terms.

No prices, quantities or NAV levels are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_decompose import GROUPS
from forward_test import FORWARD_START, ForwardPipeline, Frozen, forbidden_in, forward_paths, load_selection


def group_sums(values: np.ndarray, assets: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for a, v in zip(assets, values, strict=True):
        g = GROUPS.get(a, "other")
        out[g] = out.get(g, 0.0) + float(v)
    return out


def shares(values: np.ndarray, assets: list[str]) -> dict[str, float] | None:
    """Group shares of a risky-asset vector, renormalised to sum to one (None when the vector is empty)."""
    total = float(np.sum(values))
    if total <= 1e-12:
        return None
    return {g: round(s / total, 4) for g, s in group_sums(np.asarray(values, dtype=float), assets).items()}


def binding_term(b_prop: float, b_def: float, cash_floor_cap: float, b_final: float, tol: float = 1e-9) -> str:
    """Which of the three caps equals the final budget (the smallest; ties are reported together)."""
    names = [n for n, v in (("proposed", b_prop), ("defensive_envelope", b_def), ("cash_floor", cash_floor_cap)) if abs(v - b_final) <= tol]
    return "+".join(names) if names else "zero_mixture"


def mean_shares(rows: list[dict[str, float] | None]) -> dict[str, float]:
    rows = [r for r in rows if r]
    groups = sorted({g for r in rows for g in r})
    return {g: round(float(np.mean([r.get(g, 0.0) for r in rows])), 4) for g in groups} if rows else {}


def agent_view(p: ForwardPipeline, selection: Frozen, start_date: str = FORWARD_START) -> dict[str, Any]:
    run, block, _start, _end = forward_paths(p, selection, start_date)
    cache: dict[int, list] = {}
    for key, (tgt, ctx) in p.ev.engine.caches.targets.items():
        cache.setdefault(key[1], []).append((tgt, ctx))
    positions = set(block.decision_positions)
    decs = sorted((d for d in run.result.decisions if d.position in positions), key=lambda d: d.position)
    cash_cap = 1.0 - p.cfg.constraints.target_cash_floor
    rows, agent_names = [], []
    for d in decs:
        found = cache.get(d.position, [])
        if len(found) != 1:
            raise ValueError(f"position {d.position}: {len(found)} cached targets, expected one")
        tgt, ctx = found[0]
        assets = list(ctx.assets)
        if not np.allclose(tgt.a_bar, d.a_bar, atol=1e-12):
            raise ValueError(f"position {d.position}: the cached target differs from the decision record")
        c = tgt.combined
        agents = {}
        for o in tgt.outputs:
            agent_names.append(o.agent_name)
            agents[o.agent_name] = {"confidence": round(float(o.confidence), 4), "proposed_risky_budget": round(float(o.proposed_risky_budget), 4),
                                    "regime_gate": round(float(o.regime_gate), 4),
                                    "score_shares": shares(np.array([o.normalized_score[a] for a in assets]), assets)}
        n = len(assets)
        w_out, a_bar, w_minus = np.array(d.weights), np.array(d.a_bar), np.array(d.w_minus)
        rows.append({
            "date": d.cutoff_utc[:10], "branch": d.branch,
            "regime_probabilities": [round(float(x), 4) for x in tgt.regime_probabilities],
            "gates_raw": [round(float(x), 4) for x in c.gates_raw],
            "gates_normalised": None if c.gates_normalized is None else [round(float(x), 4) for x in c.gates_normalized],
            "abstaining": list(c.abstaining),
            "agents": agents,
            "mixture_shares": shares(np.array(c.q_bar), assets),
            "risk_pressure": round(float(c.risk_pressure), 4), "pressure_components": {k: round(float(v), 4) for k, v in c.diagnostics["pressure_components"].items()},
            "b_proposed": round(float(c.b_proposed), 4), "b_defensive": round(float(c.b_defensive), 4), "b_final": round(float(c.b_final), 4),
            "binding": binding_term(c.b_proposed, c.b_defensive, cash_cap, c.b_final),
            "cap_reduction": round(float(c.diagnostics["cap_reduction"]), 4),
            "nav_share_by_group": {k: {g: round(s, 4) for g, s in group_sums(v[:n], assets).items()}
                                   for k, v in (("target_before_optimiser", a_bar), ("optimiser_weights", w_out), ("holdings_before_trade", w_minus))},
            "sleeve_shares": {"target_before_optimiser": shares(a_bar[:n], assets), "optimiser_weights": shares(w_out[:n], assets), "holdings_before_trade": shares(w_minus[:n], assets)},
            "optimiser_shift_half_l1": round(0.5 * float(np.abs(w_out - a_bar).sum()), 4),
            "intended_one_way_turnover": round(float(d.one_way_turnover_intended), 4),
            "objective_terms_track_risk_cost": None if d.objective_terms is None else [round(float(x), 6) for x in d.objective_terms],
        })
    names = sorted(set(agent_names), key=agent_names.index)
    binding_counts: dict[str, int] = {}
    for r in rows:
        binding_counts[r["binding"]] = binding_counts.get(r["binding"], 0) + 1
    summary = {
        "decisions": len(rows), "binding_term_counts": binding_counts,
        "mean_gates_normalised": {nm: round(float(np.mean([r["gates_normalised"][i] for r in rows if r["gates_normalised"]])), 4) for i, nm in enumerate(names)} if rows else {},
        "mean_regime_probabilities": [round(float(x), 4) for x in np.mean([r["regime_probabilities"] for r in rows], axis=0)] if rows else [],
        "mean_agent_score_shares": {nm: mean_shares([r["agents"][nm]["score_shares"] for r in rows]) for nm in names},
        "mean_agent_proposed_budget": {nm: round(float(np.mean([r["agents"][nm]["proposed_risky_budget"] for r in rows])), 4) for nm in names},
        "mean_sleeve_shares": {k: mean_shares([r["sleeve_shares"][k] for r in rows]) for k in ("target_before_optimiser", "optimiser_weights", "holdings_before_trade")},
        "mean_mixture_shares": mean_shares([r["mixture_shares"] for r in rows]),
        "mean_nav_share_by_group": {k: {g: round(float(np.mean([r["nav_share_by_group"][k].get(g, 0.0) for r in rows])), 4)
                                        for g in sorted({g for r in rows for g in r["nav_share_by_group"][k]})}
                                    for k in ("target_before_optimiser", "optimiser_weights", "holdings_before_trade")},
        "mean_optimiser_shift_half_l1": round(float(np.mean([r["optimiser_shift_half_l1"] for r in rows])), 4) if rows else None,
        "mean_risk_pressure": round(float(np.mean([r["risk_pressure"] for r in rows])), 4) if rows else None,
    }
    return {"note": "descriptive agent-level view of one path; not an evidence statement, nothing here is tuned or selected",
            "selection_trial": selection.winner_hash[:19], "agent_order": names, "summary": summary, "decisions": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--finished-run", type=Path, default=None)
    ap.add_argument("--selection-json", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-id", default="agents")
    ap.add_argument("--config", default="configs/research_8_12asset.yaml")
    ap.add_argument("--start", default=FORWARD_START)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    if (args.finished_run is None) == (args.selection_json is None):
        ap.error("give exactly one of --finished-run and --selection-json")
    selection = load_selection(args.finished_run or args.selection_json)
    p = ForwardPipeline(args.root, args.out_root, args.run_id, config=args.config, allow_synthetic=args.allow_synthetic)
    text = json.dumps(agent_view(p, selection, args.start), indent=1, sort_keys=True)
    bad = forbidden_in(text)
    if bad:
        print("forbidden wording in the summary:", bad, file=sys.stderr)
        return 2
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
