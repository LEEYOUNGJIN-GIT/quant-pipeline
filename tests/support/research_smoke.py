"""A reduced research configuration for ENGINEERING tests (never a research result).

It keeps the research structure (monthly decisions that are a subset of the model rows, the HMM-filtered lane, a PCA/gate trial space, a tail-fraction
holdout) but shrinks windows, folds, trials and bootstrap so a run fits in minutes. It is written next to the test repository, not under configs/.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE_WINDOW = ("2020-01-02", "2021-12-31")

OVERRIDES: dict[str, object] = {
    "profile.name": "research_8_12asset",
    "panel.minimum_daily_rows": 60, "panel.minimum_weekly_rows": 20,
    "covariance.lookback_rows": 60, "covariance.min_observations": 30,
    "features.volatility_window": 40, "features.residual_z_window": 40,
    "features.drawdown_window": 60, "features.min_observations": 30,
    "evaluation.outer.initial_train_rows": 168, "evaluation.outer.test_rows": 84, "evaluation.outer.step_rows": 84,
    "evaluation.outer.maximum_folds": 2, "evaluation.outer.minimum_decisions_per_test": 2, "evaluation.outer.minimum_fills_per_test": 1,
    "evaluation.outer.minimum_scored_nav_intervals_per_test": 20,
    "evaluation.inner.folds": 2, "evaluation.inner.validation_rows": 63, "evaluation.inner.minimum_train_rows": 63,
    "evaluation.inner.minimum_decisions_per_validation": 2, "evaluation.inner.minimum_fills_per_validation": 1,
    "evaluation.inner.minimum_scored_nav_intervals_per_validation": 20,
    "evaluation.max_parameter_trials": 4,
    "final_holdout.minimum_rows": 42, "final_holdout.minimum_decisions": 2, "final_holdout.minimum_fills": 1,
    "final_holdout.minimum_scored_nav_intervals": 20,
    "bootstrap.intervals_are_inferential": False, "bootstrap.replicates": 100, "bootstrap.mean_block_length": 5, "bootstrap.minimum_oos_rows": 40,
    "compute_budget.max_parameter_trials": 4,
}


def _set(doc: dict, dotted: str, value: object) -> None:
    node = doc
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(dotted)
    node[parts[-1]] = value


def write_smoke_config(dest: Path, overrides: dict[str, object] | None = None) -> str:
    """Write the reduced profile to ``dest/configs/research_smoke.yaml``; returns its path relative to ``dest``."""
    doc = yaml.safe_load((ROOT / "configs" / "research_8_12asset.yaml").read_text(encoding="utf-8"))
    for k, v in {**OVERRIDES, **(overrides or {})}.items():
        if k == "compute_budget.max_parameter_trials":
            continue  # not a profile key; kept above only to document that the trial cap is the evaluation one
        _set(doc, k, v)
    (dest / "configs").mkdir(exist_ok=True)
    (dest / "configs" / "research_smoke.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return "configs/research_smoke.yaml"
