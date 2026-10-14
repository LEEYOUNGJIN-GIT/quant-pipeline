"""Terminology scan negative controls (INV-DOC-002): forbidden claims and missing labels must be caught."""
from __future__ import annotations

import pytest

from quant_pipeline.reporting import FORBIDDEN_CLAIMS, REQUIRED_PHRASES, scan_terminology

GOOD = "\n".join(p for p, _ in REQUIRED_PHRASES)


def test_a_text_with_every_required_label_and_no_claim_is_clean():
    assert scan_terminology(GOOD) == {"forbidden": [], "missing": []}


@pytest.mark.parametrize("claim", ["This is production-ready.", "ABM results", "an agent-based market simulation", "a profitable strategy",
                                   "statistically significant", "it proves the edge", "guaranteed", "it will outperform",
                                   "institutional adoption"])
def test_every_forbidden_claim_is_detected(claim):
    assert scan_terminology(GOOD + "\n" + claim)["forbidden"]


@pytest.mark.parametrize("phrase", [p for p, _ in REQUIRED_PHRASES])
def test_removing_any_required_phrase_is_reported(phrase):
    assert scan_terminology(GOOD.replace(phrase, ""))["missing"]


def test_markup_does_not_hide_a_claim():
    assert scan_terminology(GOOD + "<b>profitab</b>le")["forbidden"]


def test_every_forbidden_pattern_has_a_negative_control_above():
    assert len(FORBIDDEN_CLAIMS) == 8
