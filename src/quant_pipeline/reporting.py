"""Report rendering (spec 17, 21 item 23): Jinja2 with ``StrictUndefined`` so a missing variable fails, required disclosures and
labels present, unsupported claims absent. The pre-holdout report is complete before the holdout can be opened."""
from __future__ import annotations

import platform
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from . import SPEC_VERSION
from .benchmarks import COMPARABILITY, LABELS
from .config import Config
from .evaluation import BENCHMARK_IDS, Evaluator, OuterRun, PassOutcome, interval_returns, net_excess_sharpe

TEMPLATES = Path(__file__).parent / "templates"

# phrases a report must never contain: they assert more than a fixture can support
FORBIDDEN_CLAIMS = [
    (r"\bABM\b", "ABM shorthand without an agent-based market simulation"),
    (r"agent-based (market )?(model|simulation)", "agent-based market claim"),
    (r"profitab", "profitability claim"),
    (r"statistically significant", "significance claim"),
    (r"\bproduction[- ]ready\b", "production-readiness claim"),
    (r"\b(proves?|proven|guarantee[sd]?)\b", "proof / guarantee claim"),
    (r"\boutperform", "performance-superiority claim"),
    (r"institutional adoption", "institutional adoption claim"),
]
REQUIRED_PHRASES = [
    ("Tax is a zero placeholder", "conspicuous zero-tax label"),
    ("not production-realistic", "net results are not production-realistic"),
    ("implementation-matched", "matched-benchmark label"),
    ("initial turnover exempt", "reference label"),
    ("reference-only", "reference comparability warning"),
    ("non-inferential", "bootstrap is not inferential"),
    ("approximation", "daily-panel limitation"),
    ("Final holdout", "holdout status line"),
    ("KRW cash reference", "cash reference label"),
    ("Degraded conditions", "degraded conditions section"),
    ("multiple-trial warning", "selection-bias warning for the searched candidates"),
]


def max_drawdown(values: list[float]) -> float:
    peak, worst = -np.inf, 0.0
    for v in values:
        peak = max(peak, v)
        worst = max(worst, (peak - v) / peak) if peak > 0 else worst
    return float(worst)


def series_metrics(nav: dict[str, Decimal], sessions: list[str], rf: dict[str, float], ppy: int, costs: Decimal) -> dict[str, Any]:
    rets = interval_returns(nav)
    r = np.array([rets[s] for s in sessions])
    f = np.array([rf[s] for s in sessions])
    keys = list(nav)
    start = keys.index(sessions[0]) - 1
    path = [float(nav[k]) for k in keys[max(start, 0): keys.index(sessions[-1]) + 1]]
    sharpe = net_excess_sharpe(r, f, ppy)
    return {"total_return": float(np.prod(1.0 + r) - 1.0), "sharpe": "n/a" if not np.isfinite(sharpe) else f"{sharpe:.2f}",
            "max_drawdown": max_drawdown(path), "costs": f"{costs:,.2f}", "intervals": len(r)}


def merge_nav(runs: list[OuterRun], bench: str | None = None) -> tuple[dict[str, Decimal], Decimal]:
    nav: dict[str, Decimal] = {}
    for r in runs:
        nav.update(r.result.nav if bench is None else r.benchmarks[bench].nav)
    last = runs[-1].result if bench is None else runs[-1].benchmarks[bench]
    return nav, last.costs


def degraded_conditions(outcomes: list[PassOutcome], lineage: dict[str, str], extra: list[str] | None = None, fixture: bool = True,
                        synthetic: bool = False) -> list[str]:
    out = []
    for o in outcomes:
        bad = [e for e in o.evaluations if not e.eligible]
        for e in bad:
            out.append(f"Selection pass {o.name}: trial {e.trial_hash[7:13]} (inner fold {e.inner_index}) is below the {'fixture' if fixture else 'configured'} minima "
                       f"({e.ineligible_reason}: {e.realized_fills} realized fills, {e.executable_decisions} decisions) and is ineligible for that pass.")
        if o.selection_unstable:
            out.append(f"Selection pass {o.name} is labelled selection_unstable: leave-one-fold-out winners differ.")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        out.append(f"Runtime is CPython {platform.python_version()}, not the specification's 3.12 target.")
    if lineage["lock_hash"].startswith("unavailable"):
        out.append("No accepted dependency lock exists; the environment is not frozen.")
    if fixture:
        out.append("Decisions use the fixed fixture regime table, not the filtered hidden-Markov lane.")
    elif synthetic:
        out.append("The input snapshot is synthetic: this run is an engineering test on a reduced configuration and is not research evidence.")
    out.extend(extra or [])
    return out


def build_context(cfg: Config, ev: Evaluator, outcomes: list[PassOutcome], outer: dict[float, list[OuterRun]], summary: dict[str, Any],
                  lineage: dict[str, str], run_id: str, holdout: dict[str, Any] | None = None,
                  extra_degraded: list[str] | None = None, evidence_class: str | None = None) -> dict[str, Any]:
    ppy = cfg.evaluation.periods_per_year
    rf = ev.risk_free()
    sessions = [s for o in ev.manifest.outer for s in o.test.valuation_sessions]
    runs1 = outer[1.0]
    nav, costs = merge_nav(runs1)
    strategy = {"name": "strategy", "label": "heterogeneous-agent strategy (headline, 1x)",
                **series_metrics(nav, sessions, rf, ppy, costs)}
    bench_rows = {}
    for b in BENCHMARK_IDS:
        bn, bc = merge_nav(runs1, b)
        bench_rows[b] = {"name": b, "label": f"{LABELS[b]} [{COMPARABILITY[b]}]", **series_metrics(bn, sessions, rf, ppy, bc)}
    sens = []
    for m in sorted(outer):
        n, c = merge_nav(outer[m])
        sens.append({"name": f"{m:g}x costs", **series_metrics(n, sessions, rf, ppy, c)})
    stats = summary["statistics"]
    rows = []
    for k in ("strategy", *BENCHMARK_IDS):
        s = stats[k]
        und = s.get("undefined", 0.0) == 1.0
        fmt = (lambda x: "undefined") if und else (lambda x: f"{x:.2f}")
        rows.append({"name": k, "point": fmt(s["point_sharpe"]), "low": fmt(s["ci_low"]), "high": fmt(s["ci_high"])})
    b = cfg.bootstrap
    n_trials = len(ev.design.trials)
    last = max(ev.manifest.path_endpoints())
    inner_folds = sum(len(p.inner) for p in ev.manifest.passes)
    budget_numbers = {"evaluations": n_trials * inner_folds, "passes": len(ev.manifest.passes), "inner_folds": cfg.evaluation.inner.folds,
                      "candidates": n_trials, "executed_paths": n_trials, "executed_decisions": n_trials * last,
                      "no_sharing_decisions": n_trials * sum(ev.manifest.path_endpoints())}
    sel = []
    for o in outcomes:
        elig = sum(e.eligible for e in o.evaluations)
        agree = len(set(o.leave_one_fold_out_winners)) == 1
        sel.append({"name": o.name, "train_end": o.train_end, "winner": o.winner_hash[7:15], "margin": o.selection.margin,
                    "tie_path": " > ".join(o.selection.tie_path), "loo_agrees": "yes" if agree else "no", "unstable": o.selection_unstable,
                    "eligible": elig, "total": len(o.evaluations)})
    h = holdout or {}
    opened = bool(h.get("opened"))
    ref_run = runs1[0].benchmarks["equal_weight_buy_and_hold_reference"]
    fixture = cfg.profile.profile_type == "fixture"
    synthetic = evidence_class == "synthetic_engineering_test"
    if fixture:
        title = "Heterogeneous-agent portfolio model - fixture evaluation report"
        banner = "FIXTURE SMOKE (synthetic data); DEGRADED conditions are listed in section 8"
        data_note = ("Synthetic fixture data. Results demonstrate implementation properties only; they make no claim about any market, "
                     "about future returns, or about statistical strength.")
        limitation_regime = ("The fixture decision lane uses a fixed regime table; the hidden-Markov path is exercised only by an integration "
                             "probe against an independent oracle.")
        limitation_data = "Data are synthetic. No statement here concerns real markets."
        bootstrap_note = "the fixture bootstrap is a deterministic smoke test; its intervals are not inferential and the holdout is excluded."
        schedule = "every eligible canonical cutoff (fixture testing profile, not an economic recommendation)"
        outer_description = f"two outer folds of {cfg.evaluation.outer.test_rows} decisions after {cfg.evaluation.outer.initial_train_rows} training decisions"
    else:
        title = "Heterogeneous-agent portfolio model - research evaluation report" + (" (SYNTHETIC engineering test)" if synthetic else "")
        banner = ("SYNTHETIC ENGINEERING TEST (not research evidence); " if synthetic else "RESEARCH EVALUATION; ") + "DEGRADED conditions are listed in section 8"
        data_note = ("Synthetic engineering-test data on a reduced configuration. Results demonstrate implementation properties only; they make no claim "
                     "about any market, about future returns, or about statistical strength."
                     if synthetic else
                     "Results describe a historical evaluation of the supplied research snapshot only; they make no claim about future returns, "
                     "and the reported intervals carry the limits stated in section 7.")
        limitation_regime = "The decision lane uses the hidden-Markov filtered state (forward recursion only; no smoothing) fitted per refit."
        limitation_data = ("Data are synthetic. No statement here concerns real markets." if synthetic else
                           "Results depend on the supplied snapshot, its provider conventions and its license; they are not production-realistic.")
        bootstrap_note = (f"intervals are {'inferential' if b.intervals_are_inferential else 'non-inferential'} as configured; "
                          "the holdout is excluded.")
        schedule = "monthly: the first eligible canonical cutoff of each calendar month"
        outer_description = (f"{len(ev.manifest.outer)} outer folds of {cfg.evaluation.outer.test_rows} model rows after "
                             f"{cfg.evaluation.outer.initial_train_rows} training rows (decisions are monthly)")
    ctx = {
        "title": title, "data_note": data_note, "limitation_regime": limitation_regime, "limitation_data": limitation_data,
        "status_banner": banner,
        "spec_version": SPEC_VERSION, "profile": cfg.profile.name, "run_id": run_id, "lineage": lineage,
        "costs": {"commission_bps": cfg.costs.commission_bps, "half_spread_slippage_bps": cfg.costs.half_spread_slippage_bps,
                  "fx_conversion_bps": cfg.costs.fx_conversion_bps},
        "method": {"outer_description": outer_description,
                   "oos_intervals": len(sessions),
                   "inner_description": f"{cfg.evaluation.inner.folds} validation blocks of {cfg.evaluation.inner.validation_rows} rows per pass, "
                                        f"one gap row, {cfg.evaluation.purge.minimum_rows} purge rows, metric {cfg.evaluation.inner.selection_metric}",
                   "schedule": schedule,
                   "valuation_stale_limit": cfg.valuation.max_krx_stale_sessions},
        "budget": budget_numbers,
        "trial_warning": (f"Multiple-trial warning: {n_trials} parameter candidates (cap {cfg.evaluation.max_parameter_trials}) were scored in each of "
                          f"{len(ev.manifest.passes)} selection passes. The winner of a search is selected for scoring well on the same validation blocks, "
                          "so its selection-time score is optimistically biased and the outer results are not corrected for the search."),
        "selection": sel,
        "panel_matched": [strategy, bench_rows["equal_weight_periodic"], bench_rows["equal_weight_staged_entry"]],
        "panel_reference": [bench_rows["equal_weight_buy_and_hold_reference"]],
        "reference_initial_turnover": f"{ref_run.decisions[0].one_way_turnover_intended:.2f}",
        "cash": {"name": LABELS["cash_krw"], "total_return": bench_rows["cash_krw"]["total_return"]},
        "sensitivity": sens,
        "uncertainty": {"method": "stationary block bootstrap over the continuous aligned outer-OOS tuple", "block_length": b.mean_block_length,
                        "replicates": b.replicates, "shape": f"{summary['bootstrap_shape'][0]}x{summary['bootstrap_shape'][1]}",
                        "index_hash": summary["bootstrap_index_hash"],
                        "inference_label": "non-inferential" if not b.intervals_are_inferential else "inferential", "rows": rows,
                        "note": bootstrap_note},
        "holdout": {"status": "opened once (audited)" if opened else "sealed (unopened)", "opened": opened,
                    "audit_hash": h.get("audit_hash", "none"), "intervals": h.get("intervals", 0), "rows": h.get("rows", []),
                    "statement": (h.get("statement") or "The final holdout is sealed: no holdout outcome has been read, used for selection, or "
                                  "reported. It can be opened once, with an explicit acknowledgement, after this report is complete.")},
        "degraded": degraded_conditions(outcomes, lineage, extra_degraded, fixture, synthetic),
    }
    return ctx


def render(ctx: dict[str, Any]) -> tuple[str, str]:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined, autoescape=select_autoescape(["html"]),
                      trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)
    return env.get_template("report.md.j2").render(**ctx), env.get_template("report.html.j2").render(**ctx)


def scan_terminology(text: str, inferential: bool = False) -> dict[str, list[str]]:
    """Forbidden claims and required labels; the bootstrap status label must say what the profile configures (inferential or not)."""
    plain = re.sub(r"<[^>]+>", " ", text)
    forbidden = [why for pat, why in FORBIDDEN_CLAIMS if re.search(pat, plain, flags=re.IGNORECASE)]
    required = [(p, w) if p != "non-inferential" or not inferential else ("inferential", "bootstrap status label") for p, w in REQUIRED_PHRASES]
    missing = [why for phrase, why in required if phrase.lower() not in plain.lower()]
    return {"forbidden": forbidden, "missing": missing}
