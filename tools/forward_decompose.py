"""Descriptive decomposition of a forward-test path (OQ-061): where the strategy's return differs from the primary benchmark's.

Nothing is selected, tuned or compared against a rule here, and the numbers are not an evidence statement. The same frozen selection, snapshot and interval as
tools/forward_test.py are replayed (deterministically), and the stored ledger, decisions and benchmark paths are summarised as aggregates:

* exposure: the risky share of NAV at the start of each scored interval, for the strategy and the benchmark;
* an exposure-matched benchmark: the benchmark's own interval returns, re-weighted each interval by the strategy-to-benchmark exposure ratio (the rest earns the
  risk-free rate). If this path resembles the strategy, the gap to the benchmark is mostly how much was invested, not what was held;
* per-asset contribution (start-of-interval weight times the asset's mark return in KRW); dividends and trades inside an interval are not attributed, so the
  contributions are an approximation of the NAV return, and the residual is reported;
* the benchmark's worst drawdown episode and the strategy's exposure inside it;
* decisions: risky budget, branch counts and intended turnover; costs relative to average NAV.

Only ratios, dates and counts are printed; no prices, quantities or NAV levels.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_test import (
    FORWARD_START,
    PRIMARY_BENCHMARK,
    ForwardPipeline,
    Frozen,
    forbidden_in,
    forward_paths,
    load_selection,
)
from quant_pipeline.data import NoDataError, StaleDataError
from quant_pipeline.evaluation import interval_returns, net_excess_sharpe


def path_stats(r: np.ndarray, rf: np.ndarray, ppy: int) -> dict[str, float]:
    """Total return, max drawdown and net excess Sharpe of a series of simple interval returns."""
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return {"total_return": round(float(wealth[-1] - 1.0), 4), "max_drawdown": round(float(np.max(1.0 - wealth / peak)), 4),
            "sharpe": round(net_excess_sharpe(r, rf, ppy), 4)}


def exposure_matched(bench: np.ndarray, rf: np.ndarray, strat_risky: np.ndarray, bench_risky: np.ndarray) -> np.ndarray:
    """The benchmark's interval returns scaled interval by interval to the strategy's risky share; the remainder earns the risk-free rate."""
    scale = np.divide(strat_risky, bench_risky, out=np.zeros_like(strat_risky), where=bench_risky > 1e-12)
    return scale * bench + (1.0 - scale) * rf


def valuation_table(p: ForwardPipeline, result) -> dict[str, Any]:
    """Weights and KRW marks per asset at every valuation row of a path (the session is the row's anchor)."""
    assets = [a.asset_id for a in p.cfg.universe.assets]
    ccy = {a.asset_id: a.currency for a in p.cfg.universe.assets}
    rows = result.ledger_rows
    val = rows[(rows["kind"] == "valuation") & rows["order_id"].str.startswith("VAL-") & ~rows["order_id"].str.startswith("VAL-ACCR")]
    sessions = [o[len("VAL-"):] for o in val["order_id"]]
    if sessions != list(result.nav):
        raise ValueError("the ledger's valuation rows do not match the path's NAV sessions")
    nav = val["nav_krw"].astype(float).to_numpy()
    qty = np.array([[float(val[f"qty_{a}"].iloc[i]) for a in assets] for i in range(len(val))])
    marks = np.full(qty.shape, np.nan)
    for i, ts in enumerate(pd.to_datetime(val["ts"], utc=True)):
        fx = None
        for j, a in enumerate(assets):
            try:
                price = p.md.raw_mark(a, ts)
                if ccy[a] == "USD":
                    fx = fx or p.md.fx_mark(ts)
                    price = price * fx
                marks[i, j] = float(price)
            except (NoDataError, StaleDataError):  # an asset (or the FX quote) without a mark at that time carries no weight
                continue
    weights = np.nan_to_num(qty * marks, nan=0.0) / nav[:, None]
    return {"assets": assets, "sessions": sessions, "weights": weights, "marks": marks, "nav": nav, "missing_marks": int(np.isnan(marks[qty > 0]).sum())}


def contributions(tab: dict[str, Any], scored: list[str]) -> dict[str, Any]:
    """Start-of-interval weight times the asset's mark return, summed over the scored intervals (arithmetic, not compounded)."""
    pos = {s: i for i, s in enumerate(tab["sessions"])}
    k = np.array([pos[s] for s in scored])
    ret = tab["marks"][k] / tab["marks"][k - 1] - 1.0
    contrib = np.nansum(tab["weights"][k - 1] * np.nan_to_num(ret, nan=0.0), axis=0)
    risky = tab["weights"][k - 1].sum(axis=1)
    return {"per_asset": {a: round(float(c), 4) for a, c in zip(tab["assets"], contrib)}, "sum": round(float(contrib.sum()), 4), "risky_share": risky,
            "weights": tab["weights"][k - 1], "returns": np.nan_to_num(ret, nan=0.0)}


GROUPS = {"005930.KS": "korean_equity", "000660.KS": "korean_equity", "005380.KS": "korean_equity", "035420.KS": "korean_equity",
          "SPY": "us_and_em_equity_etf", "EEM": "us_and_em_equity_etf", "IEF": "treasury_etf", "TLT": "treasury_etf", "GLD": "gold_etf"}
ACTIVE = 0.02  # an interval enters sleeve statistics only when both paths hold at least this share of NAV in risky assets


def sleeve_return(w: np.ndarray, r: np.ndarray, drop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return of the risky sleeve (start weights renormalised to the risky assets, optionally without one asset) and the sleeve's NAV share per interval."""
    w = w.copy()
    if drop is not None:
        w[:, drop] = 0.0
    share = w.sum(axis=1)
    return np.divide((w * r).sum(axis=1), share, out=np.zeros(len(share)), where=share > 1e-9), share


def compound(x: np.ndarray) -> float:
    return round(float(np.prod(1.0 + x) - 1.0), 4)


def sleeve_analysis(assets: list[str], ws: np.ndarray, wb: np.ndarray, r: np.ndarray, rf: np.ndarray, arr_s: np.ndarray, arr_b: np.ndarray,
                    quarters: np.ndarray, scored: list[str], decs: list, ppy: int) -> dict[str, Any]:
    """Split the strategy-minus-benchmark gap into how much was invested (exposure) and what was held inside the risky sleeve (selection).

    With s, b the risky shares and rS, rB the sleeve returns, strategy - benchmark = (s - b)(rB - rf) + s (rS - rB) holds exactly for
    strategy = s rS + (1 - s) rf and benchmark = b rB + (1 - b) rf; the part of the actual gap that this does not reproduce (interval trades, dividends,
    cash interest) is reported as the residual."""
    rs, s = sleeve_return(ws, r)
    rb, b = sleeve_return(wb, r)
    expo_eff, sel_eff = (s - b) * (rb - rf), s * (rs - rb)
    gap = arr_s - arr_b
    both = (s > ACTIVE) & (b > ACTIVE)

    def by(x: np.ndarray, q: str) -> float:
        return round(float(x[quarters == q].sum()), 4)

    out: dict[str, Any] = {
        "identity": "strategy - benchmark = (s - b)(rB - rf) + s(rS - rB); sums are arithmetic over the scored intervals",
        "sum_of_gap": round(float(gap.sum()), 4), "exposure_effect": round(float(expo_eff.sum()), 4), "selection_effect": round(float(sel_eff.sum()), 4),
        "residual": round(float(gap.sum() - expo_eff.sum() - sel_eff.sum()), 4),
        "by_quarter": {q: {"gap": by(gap, q), "exposure_effect": by(expo_eff, q), "selection_effect": by(sel_eff, q)} for q in sorted(set(quarters))},
        "common_active_intervals": int(both.sum()),
    }
    if both.sum() > 1:
        out["sleeve_paths_on_common_intervals"] = {"strategy": path_stats(rs[both], rf[both], ppy), PRIMARY_BENCHMARK: path_stats(rb[both], rf[both], ppy)}
        out["sleeve_interval_win_share"] = round(float((rs[both] > rb[both]).mean()), 4)
        out["sleeve_mean_weight_inside_sleeve"] = {
            a: {"strategy": round(float((ws[both, i] / s[both]).mean()), 4), PRIMARY_BENCHMARK: round(float((wb[both, i] / b[both]).mean()), 4)} for i, a in enumerate(assets)}
    groups = sorted({GROUPS.get(a, "other") for a in assets})
    out["by_group"] = {}
    for g in groups:
        cols = [i for i, a in enumerate(assets) if GROUPS.get(a, "other") == g]
        out["by_group"][g] = {
            "mean_nav_share": {"strategy": round(float(ws[:, cols].sum(axis=1).mean()), 4), PRIMARY_BENCHMARK: round(float(wb[:, cols].sum(axis=1).mean()), 4)},
            "contribution": {"strategy": round(float((ws[:, cols] * r[:, cols]).sum()), 4), PRIMARY_BENCHMARK: round(float((wb[:, cols] * r[:, cols]).sum()), 4)}}
    out["leave_one_asset_out"] = {}
    for i, a in enumerate(assets):
        rs_i, s_i = sleeve_return(ws, r, i)
        rb_i, b_i = sleeve_return(wb, r, i)
        m = (s_i > ACTIVE) & (b_i > ACTIVE)
        if m.sum() > 1:
            out["leave_one_asset_out"][a] = {"intervals": int(m.sum()), "sleeve_return_strategy": compound(rs_i[m]), "sleeve_return_benchmark": compound(rb_i[m]),
                                             "selection_effect_sum": round(float((s_i[m] * (rs_i[m] - rb_i[m])).sum()), 4)}
    dates = [d.cutoff_utc[:10] for d in decs]
    table = []
    for i, d in enumerate(decs):
        hi = dates[i + 1] if i + 1 < len(decs) else "9999-12-31"
        w = np.array([dates[i] <= x < hi for x in scored])
        table.append({"date": dates[i], "branch": d.branch, "risky_budget": round(float(d.b_final), 4), "intended_one_way_turnover": round(float(d.one_way_turnover_intended), 4),
                      "orders": d.n_orders, "intervals": int(w.sum()),
                      "mean_realised_risky_share": round(float(s[w].mean()), 4) if w.any() else None,
                      "mean_benchmark_risky_share": round(float(b[w].mean()), 4) if w.any() else None,
                      "strategy_return": compound(arr_s[w]) if w.any() else None, "benchmark_return": compound(arr_b[w]) if w.any() else None})
    out["decision_table"] = table
    return out


def drawdown_episode(r: np.ndarray) -> tuple[int, int, float]:
    """Indices (first interval after the peak, trough interval) of the worst drawdown of a return series, and its depth."""
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak_val = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / peak_val
    trough = int(np.argmax(dd))
    peak = int(np.argmax(wealth[: trough + 1]))
    return peak, trough - 1, float(dd[trough])


def decompose(p: ForwardPipeline, selection: Frozen, start_date: str = FORWARD_START) -> dict[str, Any]:
    run, block, _start, _end = forward_paths(p, selection, start_date)
    series = {"strategy": run.result, PRIMARY_BENCHMARK: run.benchmarks[PRIMARY_BENCHMARK]}
    rets = {k: interval_returns(r.nav) for k, r in series.items()}
    scored = [s for s in block.valuation_sessions if s in rets["strategy"]]
    if len(scored) < 2:
        raise ValueError("fewer than two scored intervals")
    rf_all = p.ev.risk_free()
    rf = np.array([rf_all[s] for s in scored])
    arr = {k: np.array([rets[k][s] for s in scored]) for k in rets}
    ppy = p.cfg.evaluation.periods_per_year
    tabs = {k: valuation_table(p, r) for k, r in series.items()}
    con = {k: contributions(tabs[k], scored) for k in series}
    s_risky, b_risky = con["strategy"]["risky_share"], con[PRIMARY_BENCHMARK]["risky_share"]
    matched = exposure_matched(arr[PRIMARY_BENCHMARK], rf, s_risky, b_risky)
    mean_ratio = float(np.mean(s_risky) / np.mean(b_risky)) if np.mean(b_risky) > 0 else 0.0
    const_matched = mean_ratio * arr[PRIMARY_BENCHMARK] + (1.0 - mean_ratio) * rf

    def expo(x: np.ndarray) -> dict[str, float]:
        q = np.quantile(x, [0.1, 0.5, 0.9])
        return {"mean": round(float(x.mean()), 4), "p10": round(float(q[0]), 4), "median": round(float(q[1]), 4), "p90": round(float(q[2]), 4),
                "share_of_intervals_below_half": round(float((x < 0.5).mean()), 4)}

    quarters = pd.PeriodIndex(pd.to_datetime(scored), freq="Q").astype(str)
    by_q = {q: {"strategy_risky_share": round(float(s_risky[quarters == q].mean()), 4), "benchmark_risky_share": round(float(b_risky[quarters == q].mean()), 4),
                "strategy_return": round(float(np.prod(1 + arr["strategy"][quarters == q]) - 1), 4),
                "benchmark_return": round(float(np.prod(1 + arr[PRIMARY_BENCHMARK][quarters == q]) - 1), 4)} for q in sorted(set(quarters))}
    pk, tr, depth = drawdown_episode(arr[PRIMARY_BENCHMARK])
    inside = slice(pk, tr + 1)
    mask = np.zeros(len(scored), dtype=bool)
    mask[inside] = True
    top = max(con[PRIMARY_BENCHMARK]["per_asset"], key=lambda a: con[PRIMARY_BENCHMARK]["per_asset"][a])
    bsum = con[PRIMARY_BENCHMARK]["sum"]
    pos = set(block.decision_positions)
    decs = [d for d in series["strategy"].decisions if d.position in pos]
    budgets = np.array([d.b_final for d in decs])
    branches: dict[str, int] = {}
    for d in decs:
        branches[d.branch] = branches.get(d.branch, 0) + 1
    nav_mean = {k: float(np.mean(tabs[k]["nav"])) for k in tabs}
    sleeve = sleeve_analysis(tabs["strategy"]["assets"], con["strategy"]["weights"], con[PRIMARY_BENCHMARK]["weights"], con["strategy"]["returns"], rf,
                             arr["strategy"], arr[PRIMARY_BENCHMARK], np.asarray(quarters), scored, sorted(decs, key=lambda d: d.position), ppy)
    return {
        "sleeve_analysis": sleeve,
        "note": "descriptive decomposition of one path; not an evidence statement, nothing here is tuned or selected",
        "intervals": len(scored), "first_session": scored[0], "last_session": scored[-1], "selection_trial": selection.winner_hash[:19],
        "missing_marks": {k: tabs[k]["missing_marks"] for k in tabs},
        "paths": {"strategy": path_stats(arr["strategy"], rf, ppy), PRIMARY_BENCHMARK: path_stats(arr[PRIMARY_BENCHMARK], rf, ppy),
                  "benchmark_at_strategy_exposure_each_interval": path_stats(matched, rf, ppy),
                  "benchmark_at_strategy_mean_exposure": dict(path_stats(const_matched, rf, ppy), exposure_ratio=round(mean_ratio, 4))},
        "risky_share": {"strategy": expo(s_risky), PRIMARY_BENCHMARK: expo(b_risky)},
        "by_quarter": by_q,
        "contribution_by_asset": {"strategy": con["strategy"]["per_asset"], PRIMARY_BENCHMARK: con[PRIMARY_BENCHMARK]["per_asset"]},
        "contribution_sum_vs_arithmetic_return": {k: {"attributed": con[k]["sum"], "interval_returns_sum": round(float(arr[k].sum()), 4)} for k in con},
        "benchmark_largest_contributor": {"asset": top, "share_of_attributed_sum": round(float(con[PRIMARY_BENCHMARK]["per_asset"][top] / bsum), 4) if bsum else None,
                                          "attributed_sum_without_it": round(float(bsum - con[PRIMARY_BENCHMARK]["per_asset"][top]), 4)},
        "benchmark_worst_drawdown": {"first_interval": scored[pk], "trough_interval": scored[tr], "depth": round(depth, 4),
                                     "strategy_risky_share_inside": round(float(s_risky[mask].mean()), 4),
                                     "strategy_risky_share_outside": round(float(s_risky[~mask].mean()), 4) if (~mask).any() else None,
                                     "strategy_return_inside": round(float(np.prod(1 + arr["strategy"][mask]) - 1), 4),
                                     "benchmark_return_inside": round(float(np.prod(1 + arr[PRIMARY_BENCHMARK][mask]) - 1), 4)},
        "decisions": {"count": len(decs), "branches": branches,
                      "risky_budget": {"mean": round(float(budgets.mean()), 4), "min": round(float(budgets.min()), 4), "max": round(float(budgets.max()), 4)} if len(decs) else None,
                      "mean_intended_one_way_turnover": round(float(np.mean([d.one_way_turnover_intended for d in decs])), 4) if decs else None},
        "costs_over_mean_nav": {k: round(float(Decimal(series[k].costs)) / nav_mean[k], 6) for k in series},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--finished-run", type=Path, default=None)
    ap.add_argument("--selection-json", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-id", default="decompose")
    ap.add_argument("--config", default="configs/research_8_12asset.yaml")
    ap.add_argument("--start", default=FORWARD_START)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    if (args.finished_run is None) == (args.selection_json is None):
        ap.error("give exactly one of --finished-run and --selection-json")
    selection = load_selection(args.finished_run or args.selection_json)
    p = ForwardPipeline(args.root, args.out_root, args.run_id, config=args.config, allow_synthetic=args.allow_synthetic)
    text = json.dumps(decompose(p, selection, args.start), indent=1, sort_keys=True)
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
