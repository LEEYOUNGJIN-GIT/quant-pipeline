"""Independent verification of a finished run directory: what a user receives is intact, from one lineage, consistently labelled and sealed.

It reads only files (no pipeline code path is re-run), so it can check an artifact downloaded from elsewhere. Every check reports ok/detail; nothing is
repaired.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import hash_json_obj, sha256_bytes

HOLDOUT_MARK = "fold_id=holdout"
PRE_REPORT = ("fold_id=global", "trial_hash=pre-holdout")
POST_REPORT = ("fold_id=global", "trial_hash=post-holdout")
REQUIRED_STAGES = ("01_config_and_manifest", "10_event_time_execution_and_accounting", "11_oos_evaluation", "12_bootstrap_and_reports", "13_verification")
EVIDENCE_FILES = ("summary.json", "hmm_fits.json", "fixture_coverage.json")


def _checkpoints(run: Path) -> list[Path]:
    base = run / "checkpoints"
    return sorted(p.parent for p in base.rglob("checkpoint.json") if not any(part.startswith(".") for part in p.relative_to(base).parts))


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check_run(run: Path) -> dict[str, Any]:
    run = Path(run)
    cps = _checkpoints(run)
    checks: list[dict[str, Any]] = [_check("checkpoints_present", bool(cps), f"{len(cps)} checkpoints")]
    bad: list[str] = []
    lineages: set[str] = set()
    stages: set[str] = set()
    for d in cps:
        body = json.loads((d / "checkpoint.json").read_text(encoding="utf-8"))
        recorded = body.pop("content_hash", None)
        rel = d.relative_to(run / "checkpoints").as_posix()
        if hash_json_obj(body) != recorded or not (d / "COMPLETE").exists() or (d / "COMPLETE").read_text(encoding="utf-8") != recorded:
            bad.append(f"{rel}: checkpoint record or completion marker")
        for name, h in body["artifacts"].items():
            f = d / name
            if not f.is_file() or sha256_bytes(f.read_bytes()) != h:
                bad.append(f"{rel}/{name}")
        extra = {p.name for p in d.iterdir() if p.is_file()} - set(body["artifacts"]) - {"checkpoint.json", "COMPLETE"}
        if extra:
            bad.append(f"{rel}: unlisted files {sorted(extra)}")
        lineages.add(json.dumps(body["lineage"], sort_keys=True))
        stages.add(body["key"]["stage"])
    checks.append(_check("every_checkpoint_intact", not bad, "all hashes and markers match" if not bad else "; ".join(bad[:5]) + (f" (+{len(bad) - 5})" if len(bad) > 5 else "")))
    checks.append(_check("single_lineage", len(lineages) == 1, f"{len(lineages)} distinct lineage record(s)"))
    missing = [s for s in REQUIRED_STAGES if s not in stages]
    checks.append(_check("required_stages_present", not missing, "all present" if not missing else f"missing {missing}"))

    def find(parts: tuple[str, ...], name: str) -> Path | None:
        hits = [d / name for d in cps if all(p in d.as_posix() for p in parts) and (d / name).exists()]
        return hits[0] if hits else None

    scan = find(PRE_REPORT, "terminology_scan.json")
    scan_ok = scan is not None and json.loads(scan.read_text(encoding="utf-8")) == {"forbidden": [], "missing": []}
    checks.append(_check("pre_holdout_report_passes_terminology_scan", scan_ok, "clean" if scan_ok else "missing or not clean"))
    ctx_file, md_file, html_file = find(PRE_REPORT, "report_context.json"), find(PRE_REPORT, "report.md"), find(PRE_REPORT, "report.html")
    ctx = json.loads(ctx_file.read_text(encoding="utf-8")) if ctx_file else {}
    both = md_file is not None and html_file is not None
    checks.append(_check("report_in_both_formats", both, "report.md and report.html present" if both else "a format is missing"))

    classes: dict[str, str] = {}
    for fname in EVIDENCE_FILES:
        f = next((d / fname for d in cps if (d / fname).exists()), None)
        if f is not None:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and "evidence_class" in doc:
                classes[fname] = doc["evidence_class"]
    banner = str(ctx.get("status_banner", ""))
    synthetic = "SYNTHETIC" in banner
    consistent = len(set(classes.values())) <= 1 and (not classes or (next(iter(classes.values())) == "synthetic_engineering_test") == synthetic)
    checks.append(_check("evidence_class_consistent", consistent, f"{classes}, banner synthetic={synthetic}"))

    holdout_dirs = [d for d in cps if HOLDOUT_MARK in d.as_posix()]
    audit = next((d / "holdout_audit.json" for d in cps if (d / "holdout_audit.json").exists()), None)
    if audit is None:
        checks.append(_check("holdout_sealed", not holdout_dirs, "no holdout artifact exists" if not holdout_dirs else f"{len(holdout_dirs)} holdout artifacts without an audit record"))
    else:
        a = json.loads(audit.read_text(encoding="utf-8"))
        ok = (a.get("pre_holdout_unchanged") is True and a.get("pre_holdout_tree_hash_before") == a.get("pre_holdout_tree_hash_after")
              and a.get("opening_number") == 1 and find(POST_REPORT, "report.md") is not None)
        checks.append(_check("holdout_opened_once_with_unchanged_pre_holdout_state", ok, f"audit {str(a.get('audit_hash'))[:19]}"))
    return {"run": run.name, "checks": checks, "passed": all(c["ok"] for c in checks)}


def _artifact_bytes(run: Path) -> dict[str, bytes]:
    base = run / "checkpoints"
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*")
            if p.is_file() and p.name not in ("checkpoint.json", "COMPLETE") and not any(x.startswith(".") for x in p.relative_to(base).parts)}


def compare_runs(a: Path, b: Path, ignore_suffixes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Byte comparison of two runs' artifacts (the checkpoint records, which embed lineage and so the code hash, are excluded)."""
    x, y = _artifact_bytes(Path(a)), _artifact_bytes(Path(b))
    only_a, only_b = sorted(set(x) - set(y)), sorted(set(y) - set(x))
    different = sorted(k for k in set(x) & set(y) if x[k] != y[k] and not k.endswith(ignore_suffixes))
    return {"same_files": not only_a and not only_b, "only_in_first": only_a[:5], "only_in_second": only_b[:5], "different": different[:10], "n_different": len(different),
            "identical": not only_a and not only_b and not different}
