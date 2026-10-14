#!/usr/bin/env python3
"""Repository specification lint (spec 17.6, check ``INV-LINT-001``).

Exit codes: 0 pass, 1 contract failure, 2 missing or parse failure, 3 linter failure. Mandatory skips fail.
Writes ``outputs/spec_lint_report.json``. This is a text/structure check: a PASS is static evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COMPANIONS = ["FINAL_SPEC_RECONSTRUCTION_PROMPT.md", "ADOPTION_MATRIX.md", "PROJECT_ALIGNMENT_REVIEW.md",
              "SPEC_RECOVERY_DECISION.md"]
FORBIDDEN = [r"by_risk_rank", r"200\s*[×x]\s*20\b", r"indices?\s+0[–-]19", r"\b20 aligned\b", r"max_solver_attempts_per_problem\W+1\b"]
PROHIBITION_CUES = re.compile(r"prohibit|forbid|MUST NOT|fails?\b|invalid|never|not\b|reject|shortcut", re.IGNORECASE)


class Fail(Exception):
    """A contract failure (exit 1)."""


class Missing(Exception):
    """A missing or unparsable input (exit 2)."""


def read(path: Path) -> str:
    if not path.exists():
        raise Missing(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def section(text: str, heading_regex: str) -> str:
    m = re.search(rf"^(#+)\s+{heading_regex}.*$", text, re.MULTILINE)
    if not m:
        raise Missing(f"heading not found: {heading_regex}")
    level = len(m.group(1))
    rest = text[m.end():]
    nxt = re.search(rf"^#{{1,{level}}}\s", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def table_rows(block: str) -> list[list[str]]:
    rows = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*-", line) and not re.match(r"^\|\s*(#|Invariant|Tier)\b", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def check(name: str, fn, results: list[dict]) -> None:
    try:
        detail = fn()
        results.append({"check": name, "status": "pass", "detail": detail})
    except Fail as e:
        results.append({"check": name, "status": "fail", "detail": str(e)})
    except Missing:
        raise


def run(spec_path: Path) -> list[dict]:
    spec = read(spec_path)
    results: list[dict] = []
    version = re.search(r"\*\*Specification version:\*\*\s*`([^`]+)`", spec)
    if not version:
        raise Missing("specification version line not found")
    ver = version.group(1)

    def check_ids() -> str:
        rows = table_rows(section(spec, r"17\.2"))
        ids = [r[1].strip("`") for r in rows if len(r) >= 4 and r[1].startswith("`INV-")]
        dups = sorted({i for i in ids if ids.count(i) > 1})
        if not ids:
            raise Fail("no check IDs parsed from 17.2")
        if dups:
            raise Fail(f"duplicate primary owners: {dups}")
        check_ids.ids = set(ids)  # type: ignore[attr-defined]
        return f"{len(ids)} unique check IDs"

    check("check_ids_unique", check_ids, results)

    def dod() -> str:
        rows = [r for r in table_rows(section(spec, r"21\.")) if len(r) >= 4 and r[0].isdigit()]
        known = getattr(check_ids, "ids", set())
        prim = [r[2].strip("`") for r in rows]
        errs = []
        for r in rows:
            for token in re.findall(r"INV-[A-Z]+-\d+", r[2]):
                if token not in known:
                    errs.append(f"DoD {r[0]}: undefined check {token}")
            if not re.fullmatch(r"`INV-[A-Z]+-\d+`", r[2]):
                errs.append(f"DoD {r[0]}: not exactly one primary check ID ({r[2]})")
        dup = sorted({p for p in prim if prim.count(p) > 1})
        if dup:
            errs.append(f"DoD primary IDs shared: {dup}")
        if sorted(int(r[0]) for r in rows) != list(range(1, len(rows) + 1)):
            errs.append("DoD numbering is not contiguous")
        if errs:
            raise Fail("; ".join(errs))
        return f"{len(rows)} DoD items, each with one defined, unique primary check"

    check("dod_ownership", dod, results)

    def profiles() -> str:
        from quant_pipeline.config import Config, load_yaml_strict
        raws = {}
        for name in ("fixture_3asset", "research_8_12asset"):
            path = ROOT / "configs" / f"{name}.yaml"
            try:
                raws[name] = load_yaml_strict(read(path))
                Config.model_validate(raws[name])
            except Missing:
                raise
            except Exception as e:
                raise Fail(f"{name}: {str(e).splitlines()[0]}") from e
            if raws[name].get("schema_version") != ver:
                raise Fail(f"{name}: schema_version {raws[name].get('schema_version')} != spec {ver}")
        a, b = raws["fixture_3asset"], raws["research_8_12asset"]
        if list(a) != list(b):
            raise Fail(f"root keys differ: {sorted(set(a) ^ set(b))}")
        check_profiles.raws = raws  # type: ignore[attr-defined]
        return f"2 profiles valid, {len(a)} identical root keys"

    check_profiles = profiles
    check("profiles_strict_schema", profiles, results)

    def profile_vs_prose() -> str:
        raws = check_profiles.raws  # type: ignore[attr-defined]
        fx, rs = raws["fixture_3asset"], raws["research_8_12asset"]
        errs = []
        if (fx["valuation"]["max_krx_stale_sessions"], rs["valuation"]["max_krx_stale_sessions"]) != (2, 6):
            errs.append("valuation stale limits not 2/6")
        if not re.search(r"is 2 in the fixture profile and 6 in the research profile", spec):
            errs.append("prose 2/6 valuation sentence missing")
        for n, p in raws.items():
            if p["availability"]["max_price_stale_sessions"] != 2 or p["panel"]["max_stale_sessions"] != 2:
                errs.append(f"{n}: availability/panel freshness must be 2")
            if p["compute_budget"]["max_solver_attempts_per_problem"] != 2:
                errs.append(f"{n}: solver attempts must be 2")
        if fx["bootstrap"]["expected_index_shape"] != [200, 22] or "`200×22`" not in spec:
            errs.append("bootstrap 200x22 mismatch")
        gates = {g["name"] for g in rs["regime"]["agent_gate_candidates"]}
        need = {"balanced_2state", "balanced_3state", "defensive_tilt_2state", "defensive_tilt_3state"}
        if not need <= gates:
            errs.append(f"research gate tables missing: {sorted(need - gates)}")
        if errs:
            raise Fail("; ".join(errs))
        return "valuation 2/6, freshness 2, solver attempts 2, bootstrap 200x22, four gate tables"

    check("profile_vs_prose", profile_vs_prose, results)

    def forbidden() -> str:
        hits = []
        for i, line in enumerate(spec.splitlines(), start=1):
            for pat in FORBIDDEN:
                if re.search(pat, line) and not PROHIBITION_CUES.search(line):
                    hits.append(f"line {i}: {pat}")
        if hits:
            raise Fail("; ".join(hits))
        return "no unquoted forbidden pattern"

    check("forbidden_patterns", forbidden, results)

    def companions() -> str:
        errs = []
        for name in COMPANIONS:
            txt = read(spec_path.parent / name)
            if "IMPLEMENTATION_SPEC.md" not in txt:
                errs.append(f"{name}: does not point to IMPLEMENTATION_SPEC.md")
            if ver not in txt:
                errs.append(f"{name}: does not state version {ver}")
        if errs:
            raise Fail("; ".join(errs))
        return f"{len(COMPANIONS)} companions point to the spec at version {ver}"

    check("companion_integrity", companions, results)

    def package_version() -> str:
        from quant_pipeline import SPEC_VERSION
        if SPEC_VERSION != ver:
            raise Fail(f"package SPEC_VERSION {SPEC_VERSION} != spec {ver}")
        return f"package SPEC_VERSION == {ver}"

    check("package_version", package_version, results)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(ROOT / "IMPLEMENTATION_SPEC.md"), type=Path)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "spec_lint_report.json"), type=Path)
    args = ap.parse_args()
    try:
        results = run(args.spec)
    except Missing as e:
        print(f"MISSING/PARSE FAILURE: {e}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 3
    failed = [r for r in results if r["status"] != "pass"]
    report = {
        "linter": "tools/spec_lint.py",
        "spec_sha256": "sha256:" + hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "exit_code": 1 if failed else 0,
        "checks": results,
        "scope_note": "static structure/text checks only; not behavioral evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        print(f"{r['status'].upper():4} {r['check']}: {r['detail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
