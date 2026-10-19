"""Aggregate summary of an OPENED research run directory, safe for a public log (no prices, weights or per-date values).

Prints the audit facts (opening number, hash, whether the pre-holdout state was unchanged, whether the declared minima were met) and the holdout section of
the post-holdout report: per series total return, Sharpe and maximum drawdown over the holdout. These come from one pass through one interval and are
descriptive, not a test of anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find(run: Path, name: str, within: str | None = None) -> Path | None:
    hits = sorted(p for p in run.rglob(name) if within is None or within in p.as_posix())
    return hits[0] if hits else None


def holdout_section(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith("## 7. Holdout status")), None)
    if start is None:
        return []
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def main(argv: list[str]) -> int:
    run = Path(argv[1])
    audit_path = find(run, "holdout_audit.json")
    if audit_path is None:
        print("no holdout_audit.json: this run directory was not opened")
        return 1
    a = json.loads(audit_path.read_text(encoding="utf-8"))
    facts = {k: a.get(k) for k in ("opening_number", "max_openings_per_lineage", "pre_holdout_unchanged", "refit_policy")}
    facts["audit_hash"] = str(a.get("audit_hash"))[:19]
    facts["minima"] = a.get("minima")
    print(json.dumps({"run": run.name, **facts}, indent=1, default=str))
    report = find(run, "report.md", within="trial_hash=post-holdout")
    if report is None:
        print("no post-holdout report.md")
        return 1
    section = holdout_section(report.read_text(encoding="utf-8"))
    print("\n".join(section) if section else "the post-holdout report has no holdout section")
    return 0 if section else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
