"""Optimiser view of the forward path's decisions (OQ-063): what the optimiser's three terms and the risk model did to the agents' target.

Nothing is selected, tuned or tested against a rule, and the numbers are not an evidence statement. The same frozen selection, snapshot and interval as
tools/forward_test.py are replayed. For each decision the optimiser's inputs are rebuilt exactly as the engine built them (target ``a_bar``, holdings before
the trade, the fold's covariance, the calibrated track weights and scales, the cost rates, the constraints) and are used for three things:

* the objective's three terms, evaluated with the engine's own ``objective_terms`` at the target ``a_bar`` and at the optimiser's weights;
* the risk model: each asset group's share of portfolio variance at the target and at the weights, the stand-alone volatility of each group relative to the
  median asset, and the volatility per unit of risky exposure;
* a one-step sensitivity: the same inputs solved again with one coefficient changed to each of its candidates (and a tracking-focused corner), reported as
  group shares and the risky share of NAV. This is a property of one solve on recorded inputs, not a path: no return, Sharpe or ledger is computed from it, and
  it is not a basis for choosing a coefficient.

The weights re-solved at the selected coefficients are compared with the recorded weights; the comparison is printed so a reader knows whether the
sensitivity rests on a faithful reconstruction. No prices, quantities or NAV levels are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_agents import group_sums, mean_shares, shares
from forward_decompose import GROUPS
from forward_test import FORWARD_START, ForwardPipeline, Frozen, forbidden_in, forward_paths, load_selection
from quant_pipeline.objective import objective_terms
from quant_pipeline.optimizer import OptimizerInputs, optimize
from quant_pipeline.trials import to_trial_params


def risk_shares(w_risky: np.ndarray, sigma: np.ndarray, assets: list[str]) -> dict[str, float] | None:
    """Each group's share of portfolio variance, w_i (Sigma w)_i / (w' Sigma w), summed within the group."""
    var = float(w_risky @ sigma @ w_risky)
    if var <= 1e-18:
        return None
    contrib = w_risky * (sigma @ w_risky) / var
    return {g: round(v, 4) for g, v in group_sums(contrib, assets).items()}


def volatility_ratios(sigma: np.ndarray, assets: list[str]) -> dict[str, float]:
    """Mean stand-alone volatility of each group divided by the median asset volatility."""
    vol = np.sqrt(np.diag(sigma))
    med = float(np.median(vol))
    out: dict[str, list[float]] = {}
    for a, v in zip(assets, vol, strict=True):
        out.setdefault(GROUPS.get(a, "other"), []).append(float(v) / med)
    return {g: round(float(np.mean(v)), 3) for g, v in out.items()}


def sleeve_volatility(w_risky: np.ndarray, sigma: np.ndarray) -> float | None:
    """Volatility of the risky sleeve per unit of risky exposure, relative units (sqrt(w' Sigma w) / sum w)."""
    g = float(w_risky.sum())
    return None if g <= 1e-12 else float(np.sqrt(max(float(w_risky @ sigma @ w_risky), 0.0)) / g)


def optimiser_view(p: ForwardPipeline, selection: Frozen, start_date: str = FORWARD_START) -> dict[str, Any]:
    run, block, _start, _end = forward_paths(p, selection, start_date)
    eng, cfg = p.ev.engine, p.cfg
    trial = to_trial_params(cfg, selection.winner_parameters)
    cache = {key[1]: (tgt, ctx) for key, (tgt, ctx) in eng.caches.targets.items()}
    decs = sorted((d for d in run.result.decisions if d.position in set(block.decision_positions)), key=lambda d: d.position)
    rates = eng.rates[1.0]
    selected = (trial.lambda_track, trial.lambda_risk, trial.lambda_cost)
    grid = {"selected": selected,
            **{f"risk_{c:g}": (trial.lambda_track, c, trial.lambda_cost) for c in cfg.objective.lambda_risk_candidates},
            **{f"track_{c:g}": (c, trial.lambda_risk, trial.lambda_cost) for c in cfg.objective.lambda_track_candidates},
            "tracking_corner": (max(cfg.objective.lambda_track_candidates), min(cfg.objective.lambda_risk_candidates), trial.lambda_cost)}
    rows: list[dict[str, Any]] = []
    scen: dict[str, list[dict[str, Any]]] = {k: [] for k in grid}
    prev_w: np.ndarray | None = None
    max_diff = 0.0
    for i, d in enumerate(decs):
        tgt, ctx = cache[d.position]
        assets = list(ctx.assets)
        n = len(assets)
        fit = eng.fit(d.refit_position, trial.n_components)
        cal = eng.calibration(d.refit_position, trial, fit)
        sigma = np.asarray(fit.covariance.matrix, dtype=float)
        a_bar, w_out, w_minus = np.array(d.a_bar), np.array(d.weights), np.array(d.w_minus)

        def inputs(lam: tuple[float, float, float], *, a_bar=a_bar, w_minus=w_minus, sigma=sigma, cal=cal, tgt=tgt, ctx=ctx) -> OptimizerInputs:
            return OptimizerInputs(a_bar, w_minus, sigma, cal.w_track, cal.scales, lam, rates, tgt.combined.b_final, cfg.constraints.target_cash_floor,
                                   ctx.max_weights, eng.usd, cfg.constraints.usd_exposure_limit, cfg.constraints.turnover_one_way_limit, ctx.tradable)

        parts_at = {k: [round(float(x), 4) for x in objective_terms(w, a_bar, w_minus, cal.w_track, sigma, rates, cal.scales, selected).contributions]
                    for k, w in (("target_before_optimiser", a_bar), ("optimiser_weights", w_out))}
        sleeve_t, sleeve_w = sleeve_volatility(a_bar[:n], sigma), sleeve_volatility(w_out[:n], sigma)
        var_t, var_w = float(a_bar[:n] @ sigma @ a_bar[:n]), float(w_out[:n] @ sigma @ w_out[:n])
        rows.append({
            "date": d.cutoff_utc[:10], "branch": d.branch,
            "objective_parts_track_risk_cost": parts_at,
            "objective_part_changes_track_risk_cost": [round(parts_at["optimiser_weights"][j] - parts_at["target_before_optimiser"][j], 4) for j in range(3)],
            "risk_share_by_group": {"target_before_optimiser": risk_shares(a_bar[:n], sigma, assets), "optimiser_weights": risk_shares(w_out[:n], sigma, assets)},
            "variance_ratio_weights_over_target": None if var_t <= 1e-18 else round(var_w / var_t, 4),
            "sleeve_volatility_ratio_weights_over_target": None if not sleeve_t or sleeve_w is None else round(sleeve_w / sleeve_t, 4),
            "volatility_relative_to_median_asset": volatility_ratios(sigma, assets),
        })
        for name, lam in grid.items():
            res = optimize(inputs(lam), cfg, previous_target=prev_w, first_decision=(i == 0))
            if res.weights is None:
                scen[name].append({"failed": res.status})
                continue
            w = np.asarray(res.weights, dtype=float)
            if name == "selected":
                max_diff = max(max_diff, float(np.abs(w - w_out).max()))
            scen[name].append({"sleeve": shares(w[:n], assets), "risky_nav": float(w[:n].sum()),
                               "shift": 0.5 * float(np.abs(w - a_bar).sum()), "by_group_nav": group_sums(w[:n], assets), "branch": res.branch})
        prev_w = w_out
    sens = {}
    for name, lst in scen.items():
        ok = [x for x in lst if "failed" not in x]
        sens[name] = {"coefficients_track_risk_cost": [round(float(x), 3) for x in grid[name]], "solves": len(lst), "failed": len(lst) - len(ok),
                      "mean_sleeve_shares": mean_shares([x["sleeve"] for x in ok]),
                      "mean_nav_share_by_group": {g: round(float(np.mean([x["by_group_nav"].get(g, 0.0) for x in ok])), 4) for g in sorted({g for x in ok for g in x["by_group_nav"]})} if ok else {},
                      "mean_risky_nav_share": round(float(np.mean([x["risky_nav"] for x in ok])), 4) if ok else None,
                      "mean_shift_from_target_half_l1": round(float(np.mean([x["shift"] for x in ok])), 4) if ok else None}
    mean_parts = {k: [round(float(np.mean([r["objective_parts_track_risk_cost"][k][j] for r in rows])), 4) for j in range(3)]
                  for k in ("target_before_optimiser", "optimiser_weights")}
    risk_groups = {k: mean_shares([r["risk_share_by_group"][k] for r in rows]) for k in ("target_before_optimiser", "optimiser_weights")}
    summary = {"decisions": len(rows), "selected_coefficients_track_risk_cost": [round(float(x), 3) for x in selected],
               "mean_objective_parts_track_risk_cost": mean_parts,
               "mean_part_changes_track_risk_cost": [round(mean_parts["optimiser_weights"][j] - mean_parts["target_before_optimiser"][j], 4) for j in range(3)],
               "mean_risk_share_by_group": risk_groups,
               "mean_volatility_relative_to_median_asset": mean_shares([r["volatility_relative_to_median_asset"] for r in rows]),
               "mean_variance_ratio_weights_over_target": round(float(np.mean([r["variance_ratio_weights_over_target"] for r in rows if r["variance_ratio_weights_over_target"]])), 4),
               "mean_sleeve_volatility_ratio_weights_over_target": round(float(np.mean([r["sleeve_volatility_ratio_weights_over_target"] for r in rows if r["sleeve_volatility_ratio_weights_over_target"]])), 4),
               "reconstruction_max_abs_weight_difference_at_selected": round(max_diff, 8)}
    return {"note": "descriptive optimiser-level view of one path; not an evidence statement, nothing here is tuned or selected; the sensitivity is one solve per decision on recorded inputs, not a path",
            "selection_trial": selection.winner_hash[:19], "summary": summary, "sensitivity": sens, "decisions": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--finished-run", type=Path, default=None)
    ap.add_argument("--selection-json", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-id", default="optimiser")
    ap.add_argument("--config", default="configs/research_8_12asset.yaml")
    ap.add_argument("--start", default=FORWARD_START)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    if (args.finished_run is None) == (args.selection_json is None):
        ap.error("give exactly one of --finished-run and --selection-json")
    selection = load_selection(args.finished_run or args.selection_json)
    p = ForwardPipeline(args.root, args.out_root, args.run_id, config=args.config, allow_synthetic=args.allow_synthetic)
    text = json.dumps(optimiser_view(p, selection, args.start), indent=1, sort_keys=True)
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
