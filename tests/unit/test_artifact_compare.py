"""Numeric comparison of two research runs: last-bit noise is told apart from a categorical difference (INV-DATA-001 evidence tooling)."""
from __future__ import annotations

import json
from pathlib import Path

from quant_pipeline.artifact_check import compare_numeric


def run(root: Path, files: dict[str, object]) -> Path:
    for name, body in files.items():
        p = root / "checkpoints" / "fold_id=global" / "stage=x" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return root


BASE = {"stats.json": {"sharpe": 1.356, "ci": [1.048, 1.671], "winner": "sha256:aa", "n": 3}, "report.md": "text 1.36"}


def test_identical_runs_have_no_difference(tmp_path):
    r = compare_numeric(run(tmp_path / "a", BASE), run(tmp_path / "b", BASE))
    assert r["n_differing_files"] == 0 and r["no_categorical_difference"] and r["max_rel"] == 0.0


def test_last_bit_noise_is_numeric_only_and_measured(tmp_path):
    noisy = {**BASE, "stats.json": {"sharpe": 1.356 * (1 + 3e-12), "ci": [1.048, 1.671 + 2e-12], "winner": "sha256:aa", "n": 3}}
    r = compare_numeric(run(tmp_path / "a", BASE), run(tmp_path / "b", noisy))
    assert r["n_differing_files"] == 1 and r["no_categorical_difference"]
    assert 0 < r["max_rel"] < 1e-9 and r["worst_numeric"][0][0].endswith("stats.json")


def test_a_different_selected_hash_is_categorical_whatever_the_numbers_do(tmp_path):
    other = {**BASE, "stats.json": {"sharpe": 1.356, "ci": [1.048, 1.671], "winner": "sha256:bb", "n": 3}}
    r = compare_numeric(run(tmp_path / "a", BASE), run(tmp_path / "b", other))
    assert not r["no_categorical_difference"] and any(k.endswith("stats.json") for k in r["categorical"])


def test_a_different_shape_or_key_set_is_categorical(tmp_path):
    shorter = {**BASE, "stats.json": {"sharpe": 1.356, "ci": [1.048], "winner": "sha256:aa", "n": 3}}
    assert not compare_numeric(run(tmp_path / "a", BASE), run(tmp_path / "b", shorter))["no_categorical_difference"]
    fewer = {**BASE, "stats.json": {"sharpe": 1.356, "winner": "sha256:aa", "n": 3}}
    assert not compare_numeric(run(tmp_path / "c", BASE), run(tmp_path / "d", fewer))["no_categorical_difference"]


def test_a_large_numeric_difference_is_reported_not_hidden(tmp_path):
    big = {**BASE, "stats.json": {"sharpe": 0.5, "ci": [1.048, 1.671], "winner": "sha256:aa", "n": 3}}
    r = compare_numeric(run(tmp_path / "a", BASE), run(tmp_path / "b", big))
    assert r["no_categorical_difference"] and r["max_rel"] > 0.5  # the caller decides what is tolerable; the tool never calls it equal


def test_text_artifacts_are_counted_and_a_missing_file_is_flagged(tmp_path):
    changed = {**BASE, "report.md": "text 1.37"}
    r = compare_numeric(run(tmp_path / "a", BASE), run(tmp_path / "b", changed))
    assert r["n_other"] == 1 and r["other_files"][0].endswith("report.md")
    missing = {"stats.json": BASE["stats.json"]}
    assert not compare_numeric(run(tmp_path / "c", BASE), run(tmp_path / "d", missing))["same_files"]
