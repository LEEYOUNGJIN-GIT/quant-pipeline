"""Sealed final holdout (spec 11.6): one audited opening per lineage.

Opening needs the explicit flag AND the acknowledgement environment variable. It refuses a second opening, needs the frozen
final selection and a complete pre-holdout report, runs with the frozen hyperparameters (forward refits only), writes an
immutable audit record, and proves the pre-holdout artifacts are byte-identical afterwards.
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import serialization as S
from .checkpoints import CheckpointKey
from .evaluation import BENCHMARK_IDS, OuterRun
from .hashing import hash_json_obj
from .reporting import build_context, merge_nav, render, scan_terminology, series_metrics
from .splits import OuterFold

if TYPE_CHECKING:
    from .pipeline import FixturePipeline


class HoldoutRefused(RuntimeError):
    """The opening preconditions are not met; nothing was computed and nothing was written except the attempt log."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HoldoutContamination(RuntimeError):
    """A pre-holdout artifact changed while the holdout was open: the lineage is invalid."""


def _lp(path: Path) -> Path:
    prefix = "\\\\?\\"
    s = str(Path(path).resolve())
    return Path(prefix + s) if sys.platform == "win32" and not s.startswith(prefix) else Path(s)


def tree_hash(root: Path, skip_dirs: tuple[str, ...] = ()) -> str:
    """SHA-256 over relative path + bytes of every file, excluding directories whose name contains a skipped fragment."""
    base = _lp(root)
    h = hashlib.sha256()
    for f in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = f.relative_to(base).as_posix()
        if any(s in rel for s in skip_dirs) or f.name == "progress_log.txt" or f.parent.name.startswith("."):
            continue
        h.update(rel.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def invalid_pre_holdout_checkpoints(p: FixturePipeline) -> list[str]:
    """Every checkpoint of this run outside the holdout that fails validation (missing artifact, altered bytes, wrong lineage), as ``path: reason``."""
    base = p.store.root / p.run_id / "checkpoints"
    bad: list[str] = []
    for meta in sorted(_lp(base).rglob("checkpoint.json")):
        d = meta.parent
        rel = d.relative_to(_lp(base)).as_posix()
        if "fold_id=holdout" in rel or "trial_hash=post-holdout" in rel or any(part.startswith(".") for part in Path(rel).parts):
            continue
        fold, trial, stage, refit = (part.split("=", 1)[1] for part in Path(rel).parts)
        ok, why = p.store.validate(CheckpointKey(p.run_id, fold, trial, stage, refit), p.lineage)
        if not ok:
            bad.append(f"{rel}: {why}")
    return bad


def audit_key(p: FixturePipeline) -> CheckpointKey:
    return p.key("11_oos_evaluation", "holdout", "audit")


def _attempt(p: FixturePipeline, outcome: str) -> None:
    log = p.store.root / p.run_id / "holdout_attempts.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} {outcome}\n")


def is_open(p: FixturePipeline) -> bool:
    return p.store.validate(audit_key(p), p.lineage)[0]


def open_holdout(p: FixturePipeline, *, unlock: bool, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    fh = p.cfg.final_holdout
    if not unlock:
        _attempt(p, "refused: missing --unlock-final-holdout")
        raise HoldoutRefused("missing_unlock_flag")
    if env.get(fh.acknowledgement_env) != fh.required_acknowledgement:
        _attempt(p, "refused: missing or wrong acknowledgement")
        raise HoldoutRefused("missing_or_wrong_acknowledgement")
    if is_open(p):
        _attempt(p, "refused: already opened (max_openings_per_lineage = 1)")
        raise HoldoutRefused("already_opened")
    # The pre-holdout state must already exist and be intact: opening never computes, repairs or re-derives it (a corrupt artifact would be
    # quarantined and silently recomputed by an ordinary resume, which is not acceptable for the one irreversible step).
    for name, key in (("final_selection", p.key("11_oos_evaluation", "final", "freeze")),
                      ("pre_holdout_report", p.key("12_bootstrap_and_reports", "global", "pre-holdout"))):
        if not p.store.validate(key, p.lineage)[0]:
            _attempt(p, f"refused: {name} missing or invalid")
            raise HoldoutRefused(f"{name}_missing_or_invalid")
    invalid = invalid_pre_holdout_checkpoints(p)
    if invalid:
        _attempt(p, f"refused: {len(invalid)} pre-holdout checkpoint(s) invalid, first {invalid[0]}")
        raise HoldoutRefused("pre_holdout_artifact_invalid")
    p.run()  # loads every pre-holdout checkpoint (all valid, so nothing is computed); refuses on a lineage mismatch

    skip = ("fold_id=holdout",)
    before = tree_hash(p.store.root / p.run_id / "checkpoints", skip)
    final = next(o for o in p.outcomes if o.name == "final")
    hold = OuterFold("holdout", p.manifest.passes[-1].train_end, p.manifest.holdout)
    runs: dict[float, OuterRun] = {}
    keys, parents = [], [p.key("11_oos_evaluation", "final", "freeze")]
    for mult, carry in p.final_carry.items():
        k = p.key("11_oos_evaluation", f"holdout-x{mult:g}", f"selected-{final.winner_hash[7:19]}")
        holder: dict[str, Any] = {}

        def compute(mult=mult, carry=carry, holder=holder):
            run, new_carry = p.ev.run_outer_fold(hold, final, mult, carry)
            holder["run"] = run
            return S.outer_to_artifacts(run, new_carry)

        art = p.unit(k, parents, compute, {"winner": final.winner_hash, "multiplier": mult, "sealed_until": "opened"})
        runs[mult] = S.outer_from_artifacts(art)[0]
        keys.append(k)
    after = tree_hash(p.store.root / p.run_id / "checkpoints", skip)
    if before != after:
        raise HoldoutContamination("a pre-holdout artifact changed while the holdout was open")

    rf = p.ev.risk_free()
    sessions = list(p.manifest.holdout.valuation_sessions)
    oos_nav, oos_costs = merge_nav(p.outer_runs[1.0])
    rows = []
    for name, run_ in [("strategy", runs[1.0].result)] + [(b, runs[1.0].benchmarks[b]) for b in BENCHMARK_IDS]:
        prior = oos_nav if name == "strategy" else merge_nav(p.outer_runs[1.0], name)[0]
        prior_costs = oos_costs if name == "strategy" else merge_nav(p.outer_runs[1.0], name)[1]
        nav = {**prior, **run_.nav}
        rows.append({"name": name, **series_metrics(nav, sessions, rf, p.cfg.evaluation.periods_per_year, run_.costs - prior_costs)})
    minima = {
        "decisions": len(runs[1.0].result.decisions), "intervals": len(sessions),
        "fills": len([f for f in runs[1.0].result.fills if f.status in ("filled", "reduced_cash", "reduced_holdings")]),
        "required": {"decisions": fh.minimum_decisions, "fills": fh.minimum_fills, "intervals": fh.minimum_scored_nav_intervals}}
    minima["met"] = (minima["decisions"] >= fh.minimum_decisions and minima["fills"] >= fh.minimum_fills
                     and minima["intervals"] >= fh.minimum_scored_nav_intervals)
    carried = sorted(d for d in runs[1.0].result.nav if d < sessions[0])  # valued but neither decided on nor scored (rows left over before the holdout)
    audit = {"opened_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "opening_number": 1,
             "max_openings_per_lineage": fh.max_openings_per_lineage, "acknowledgement_env": fh.acknowledgement_env,
             "acknowledgement_provided": True, "final_selection_trial": final.winner_hash, "refit_policy": fh.refit_policy,
             "pre_holdout_tree_hash_before": before, "pre_holdout_tree_hash_after": after, "pre_holdout_unchanged": True,
             "lineage": p.lineage, "holdout_checkpoints": [k.ident() for k in keys], "minima": minima,
             "split_manifest_hash": p.manifest.hash, **p.holdout_binding(), **({"unscored_carry_sessions": len(carried)} if carried else {})}
    audit["audit_hash"] = hash_json_obj({k: v for k, v in audit.items() if k != "opened_at_utc"})
    p.store.write(audit_key(p), {"holdout_audit.json": S.dumps(audit)}, lineage=p.lineage, parents=keys, meta={"audit_hash": audit["audit_hash"]})
    from .diagnostics import scope_report
    from .trials import to_trial_params

    diag = scope_report(p.ev.engine, {"holdout": list(p.manifest.holdout.decision_positions)}, to_trial_params(p.cfg, final.winner_parameters))
    p.store.write(p.key("13_verification", "holdout", "agent-diagnostics"), {"agent_pairwise_diagnostics.json": S.dumps(diag)},
                  lineage=p.lineage, parents=[audit_key(p)])
    _attempt(p, f"opened: audit {audit['audit_hash'][:19]}")

    extra = p.holdout_evidence(final, sorted({d.refit_position for d in runs[1.0].result.decisions}))
    if extra:
        p.store.write(p.key("06_hmm_filtered_state", "holdout"), extra, lineage=p.lineage, parents=[audit_key(p)])

    h = {"opened": True, "audit_hash": audit["audit_hash"], "intervals": len(sessions), "decisions": minima["decisions"], "rows": [
        {"name": r["name"], "label": r["name"], "total_return": r["total_return"], "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"]}
        for r in rows], "statement": (f"The final holdout was opened once (audit {audit['audit_hash'][:19]}) with the hyperparameters frozen "
                                      "before it was touched; forward refits used only observations available at their cutoffs. "
                                      "The pre-holdout report above is unchanged and no selection used these results.")}
    ctx = build_context(p.cfg, p.ev, p.outcomes, p.outer_runs, p.summary, p.lineage, p.run_id, holdout=h, evidence_class=p.evidence_class,
                       research=p._report_extra(p.outcomes, p.outer_runs))
    md, html = render(ctx)
    scan = scan_terminology(md + "\n" + html, inferential=p.cfg.bootstrap.intervals_are_inferential)
    p.store.write(p.key("12_bootstrap_and_reports", "global", "post-holdout"),
                  {"report.md": md.encode("utf-8"), "report.html": html.encode("utf-8"), "terminology_scan.json": S.dumps(scan)},
                  lineage=p.lineage, parents=[audit_key(p)])
    return {"audit": audit, "holdout_rows": rows, "scan": scan}
