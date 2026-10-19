"""Forward test: apply the ALREADY FROZEN final selection of a finished research run to dates after its data window.

Nothing is selected here. The winner (trial hash and parameters) is read from the finished run's ``final_selection.json``; a longer snapshot supplies
dates that the selection never saw; the strategy and the declared benchmarks run over ``[start, end of snapshot]`` from the opening holdings (cash), with
refits that use only observations available at each cutoff. The sealed holdout of the finished run is neither opened nor evaluated.

The decision rule below is fixed BEFORE any real data is looked at and is not to be changed to suit a result. It is descriptive: a verdict about one path
through one interval, never a statement that the strategy is or is not valid.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.bootstrap import percentile_interval, run_bootstrap
from quant_pipeline.evaluation import interval_returns, net_excess_sharpe
from quant_pipeline.reporting import FORBIDDEN_CLAIMS, series_metrics
from quant_pipeline.research_pipeline import ResearchPipeline
from quant_pipeline.splits import OuterFold, _block

# ---- pre-registered (implementer's choice, to be confirmed by the user; see open-questions.md) -----------------------------------------------
FORWARD_START = "2025-04-23"  # first model row on or after this date; amended from 2025-01-02 before any result existed (OQ-056): the FX series has a gap on 2025-04-18/21
PRIMARY_BENCHMARK = "equal_weight_periodic"
VERDICTS = {
    "supports": "the 95% interval of (strategy - benchmark) net excess Sharpe lies above zero",
    "no_evidence": "the 95% interval of (strategy - benchmark) net excess Sharpe contains zero",
    "against": "the 95% interval of (strategy - benchmark) net excess Sharpe lies below zero",
}


def verdict(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if lo > 0:
        return "supports"
    if hi < 0:
        return "against"
    return "no_evidence"


@dataclass(frozen=True)
class Frozen:
    """What the evaluator needs of a selection outcome."""

    winner_hash: str
    winner_parameters: dict[str, Any]


def load_selection(run_dir: Path) -> Frozen:
    hits = sorted(run_dir.rglob("final_selection.json"))
    if not hits:
        raise FileNotFoundError(f"no final_selection.json under {run_dir}")
    doc = json.loads(hits[0].read_text(encoding="utf-8"))
    if not doc.get("frozen") or not doc.get("created_before_holdout_access"):
        raise ValueError("the selection is not marked frozen before holdout access")
    return Frozen(doc["trial_hash"], doc["parameters"])


class ForwardPipeline(ResearchPipeline):
    """The research pipeline's data, calendar and evaluator wiring on a LONGER snapshot. The outer-fold ceiling guards the selection stage, which this
    test never runs, so it is lifted; nothing else of the configuration is changed."""

    def _load_config(self, path: Path):
        cfg = super()._load_config(path)
        outer = cfg.evaluation.outer.model_copy(update={"maximum_folds": 999})
        return cfg.model_copy(update={"evaluation": cfg.evaluation.model_copy(update={"outer": outer})})


def forward_start_position(p: ForwardPipeline, start_date: str) -> int:
    elig = list(p.rm.eligible)
    first = next((i for i, s in enumerate(elig) if pd.Timestamp(s).tz_localize(None) >= pd.Timestamp(start_date)), None)
    if first is None:
        raise ValueError(f"the snapshot has no model row on or after {start_date}")
    if first < 1:
        raise ValueError("the forward interval starts at the first model row: there is no history before it")
    return first + 1  # 1-based model position


def forward_paths(p: ForwardPipeline, selection: Frozen, start_date: str = FORWARD_START):
    """The strategy and benchmark paths over the forward interval, from the opening holdings (no carry from any earlier run)."""
    start = forward_start_position(p, start_date)
    end = len(p.rm.eligible)
    block = _block("forward", start, end, p.cfg, p.cals, p.rm)
    run, _ = p.ev.run_outer_fold(OuterFold("forward", start - 1, block), selection, 1.0, None)
    return run, block, start, end


def forward_run(p: ForwardPipeline, selection: Frozen, start_date: str = FORWARD_START) -> dict[str, Any]:
    run, block, start, end = forward_paths(p, selection, start_date)
    series = {"strategy": (run.result.nav, run.result.costs)} | {b: (r.nav, r.costs) for b, r in run.benchmarks.items()}
    rets = {k: interval_returns(nav) for k, (nav, _) in series.items()}
    # The engine's NAV starts at the opening timestamp, so it also holds valuations before the forward start; the block owns only its own anchors.
    sessions = [s for s in block.valuation_sessions if s in rets["strategy"]]
    if not sessions:
        raise ValueError("the forward block owns no valuation interval")
    for k, r in rets.items():
        if any(s not in r for s in sessions):
            raise ValueError(f"{k}: its valuation intervals differ from the strategy's")
    rf_all = p.ev.risk_free()
    rf = np.array([rf_all[s] for s in sessions])
    arrays = {k: np.array([r[s] for s in sessions]) for k, r in rets.items()}
    b = p.cfg.bootstrap
    boot = run_bootstrap(arrays, rf, replicates=b.replicates, mean_block_length=b.mean_block_length,
                         seed=p.cfg.reproducibility.random_streams.bootstrap, confidence=b.confidence_level,
                         periods_per_year=p.cfg.evaluation.periods_per_year, minimum_rows=b.minimum_oos_rows, expected_rows=None,
                         inferential=b.intervals_are_inferential)
    out: dict[str, Any] = {
        "evidence_class": p.evidence_class, "snapshot_id": p.snapshot.snapshot_id[:19], "selection_trial": selection.winner_hash[:19],
        "forward_start_session": str(pd.Timestamp(p.rm.eligible[start - 1]).date()), "forward_end_session": str(pd.Timestamp(p.rm.eligible[end - 1]).date()),
        "intervals": len(sessions), "decisions": block.n_decisions, "bootstrap_status": boot.status,
        "start_state": "opening holdings (cash); no continuation of the finished run's ledger",
        "rule": {"benchmark": PRIMARY_BENCHMARK, "verdicts": VERDICTS}}
    if boot.status != "ok":
        out["verdict"] = "insufficient_sample"
        return out
    ppy = p.cfg.evaluation.periods_per_year
    out["statistics"] = {k: {x: (None if not np.isfinite(v) else round(v, 4)) for x, v in st.items() if x in ("point_sharpe", "ci_low", "ci_high")}
                         for k, st in boot.statistics.items()}
    out["metrics"] = {k: series_metrics(nav, sessions, rf_all, ppy, costs if isinstance(costs, Decimal) else Decimal(costs))
                      for k, (nav, costs) in series.items() if k in ("strategy", PRIMARY_BENCHMARK)}
    idx = boot.indices
    ex = {k: arrays[k] - rf for k in ("strategy", PRIMARY_BENCHMARK)}

    def sharpe_rows(x: np.ndarray) -> np.ndarray:
        s = x[idx]
        sd = s.std(axis=1, ddof=1)
        return np.where(sd > 0, s.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(ppy), np.nan)

    diff = sharpe_rows(ex["strategy"]) - sharpe_rows(ex[PRIMARY_BENCHMARK])
    diff = diff[np.isfinite(diff)]
    lo, hi = percentile_interval(diff, b.confidence_level)
    point = net_excess_sharpe(arrays["strategy"], rf, ppy) - net_excess_sharpe(arrays[PRIMARY_BENCHMARK], rf, ppy)
    out["difference_vs_benchmark"] = {"point": round(float(point), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "replicates": int(diff.size)}
    out["verdict"] = verdict(lo, hi)
    out["verdict_meaning"] = VERDICTS.get(out["verdict"], "undefined")
    return out


def forbidden_in(text: str) -> list[str]:
    import re

    return [why for pat, why in FORBIDDEN_CLAIMS if re.search(pat, text, flags=re.IGNORECASE)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root holding data/snapshots/research/current (the LONGER snapshot)")
    ap.add_argument("--finished-run", type=Path, required=True, help="directory of the finished (decrypted) run whose frozen selection is applied")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-id", default="forward")
    ap.add_argument("--config", default="configs/research_8_12asset.yaml")
    ap.add_argument("--start", default=FORWARD_START)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    selection = load_selection(args.finished_run)
    p = ForwardPipeline(args.root, args.out_root, args.run_id, config=args.config, allow_synthetic=args.allow_synthetic)
    result = forward_run(p, selection, args.start)
    text = json.dumps(result, indent=1, sort_keys=True)
    bad = forbidden_in(text)
    if bad:
        print(f"the summary contains forbidden wording: {bad}")
        return 2
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
