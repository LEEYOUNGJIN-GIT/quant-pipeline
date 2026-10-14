"""Document contract (INV-DOC-001): open questions and progress records keep their structure; no silent drift."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OQ = (ROOT / "open-questions.md").read_text(encoding="utf-8")
PROGRESS = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")


def oq_rows():
    return [ln for ln in OQ.splitlines() if ln.startswith("| OQ-")]


def test_open_questions_have_five_columns_unique_ids_and_a_status():
    ids = []
    for ln in oq_rows():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        assert len(cells) == 5 and cells[4], ln[:60]
        ids.append(cells[0])
    assert len(ids) == len(set(ids)) and len(ids) >= 30


def test_every_oq_reference_in_progress_resolves():
    known = {ln.split("|")[1].strip() for ln in oq_rows()}
    for ref in set(re.findall(r"OQ-\d{3}", PROGRESS)):
        assert ref in known, ref


def test_deviations_are_numbered_and_the_spec_is_declared_normative():
    assert re.findall(r"^\| D(\d) ", PROGRESS, flags=re.MULTILINE) == [str(i) for i in range(1, 6)]
    assert "Normative spec: `IMPLEMENTATION_SPEC.md` v5.3.2" in PROGRESS and "never redefines the spec" in PROGRESS


def test_every_phase_declares_an_engineering_budget():
    phases = re.findall(r"^## Phase (\d)", PROGRESS, flags=re.MULTILINE)
    assert phases[:2] == ["0", "1"]
    for n in phases:
        body = PROGRESS.split(f"## Phase {n}", 1)[1].split("\n## ", 1)[0]
        assert "udget" in body, f"Phase {n} has no engineering budget"
