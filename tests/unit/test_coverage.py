"""Coverage matrix (INV-FIX-001): zero rows are reported as uncovered, never filled in."""
from __future__ import annotations

from quant_pipeline.coverage import _is_empty


def test_empty_rows_are_detected_recursively():
    assert _is_empty({}) and _is_empty({"a": 0, "b": {"c": []}}) and _is_empty(0)
    assert not _is_empty({"a": 0, "b": {"c": [1]}}) and not _is_empty({"x": "1"})
