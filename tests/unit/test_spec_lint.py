"""Negative controls for tools/spec_lint.py: a pass is only meaningful if known-bad inputs fail."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LINT = ROOT / "tools" / "spec_lint.py"
FILES = ["IMPLEMENTATION_SPEC.md", "FINAL_SPEC_RECONSTRUCTION_PROMPT.md", "ADOPTION_MATRIX.md",
         "PROJECT_ALIGNMENT_REVIEW.md", "SPEC_RECOVERY_DECISION.md"]


def run(spec_dir: Path):
    out = spec_dir / "report.json"
    p = subprocess.run([sys.executable, str(LINT), "--spec", str(spec_dir / "IMPLEMENTATION_SPEC.md"),
                        "--output", str(out)], capture_output=True, text=True, check=False)
    return p, out


@pytest.fixture
def workdir(tmp_path):
    for f in FILES:
        shutil.copy(ROOT / f, tmp_path / f)
    return tmp_path


def edit(workdir: Path, old: str, new: str, file: str = "IMPLEMENTATION_SPEC.md"):
    p = workdir / file
    t = p.read_text(encoding="utf-8")
    assert old in t, f"negative-control anchor not found: {old!r}"
    p.write_text(t.replace(old, new, 1), encoding="utf-8")


def failing(report: Path) -> set[str]:
    return {c["check"] for c in json.loads(report.read_text(encoding="utf-8"))["checks"] if c["status"] != "pass"}


def test_clean_copy_passes(workdir):
    p, out = run(workdir)
    assert p.returncode == 0, p.stdout + p.stderr
    assert failing(out) == set()


def test_duplicate_check_id_fails(workdir):
    edit(workdir, "| Fixture/research path isolation | `INV-DATA-001` |", "| Fixture/research path isolation | `INV-ENV-001` |")
    p, out = run(workdir)
    assert p.returncode == 1 and "check_ids_unique" in failing(out)


def test_dod_with_undefined_check_fails(workdir):
    edit(workdir, "| `INV-FIX-001` |", "| `INV-NOPE-999` |")
    p, out = run(workdir)
    assert p.returncode == 1 and "dod_ownership" in failing(out)


def test_dod_shared_primary_owner_fails(workdir):
    edit(workdir, "| `INV-OOS-006` |", "| `INV-OOS-002` |")
    p, out = run(workdir)
    assert p.returncode == 1 and "dod_ownership" in failing(out)


def test_unquoted_forbidden_pattern_fails(workdir):
    edit(workdir, "## 22. Canonical change control", "## 22. Canonical change control\n\nThe bootstrap uses a 200×20 index matrix.")
    p, out = run(workdir)
    assert p.returncode == 1 and "forbidden_patterns" in failing(out)


def test_quoted_prohibition_is_not_flagged(workdir):
    edit(workdir, "## 22. Canonical change control", "## 22. Canonical change control\n\n`by_risk_rank` is prohibited.")
    p, _ = run(workdir)
    assert p.returncode == 0


def test_companion_with_wrong_version_fails(workdir):
    p0 = workdir / "ADOPTION_MATRIX.md"
    p0.write_text(p0.read_text(encoding="utf-8").replace("5.3.2", "5.3.1"), encoding="utf-8")
    p, out = run(workdir)
    assert p.returncode == 1 and "companion_integrity" in failing(out)


def test_missing_companion_is_exit_2(workdir):
    (workdir / "SPEC_RECOVERY_DECISION.md").unlink()
    p, _ = run(workdir)
    assert p.returncode == 2


def test_missing_spec_is_exit_2(tmp_path):
    p, _ = run(tmp_path)
    assert p.returncode == 2


def test_missing_prose_sentence_fails_profile_vs_prose(workdir):
    edit(workdir, "is 2 in the fixture profile and 6 in the research profile", "is 2 in the fixture profile and 2 in the research profile")
    p, out = run(workdir)
    assert p.returncode == 1 and "profile_vs_prose" in failing(out)
