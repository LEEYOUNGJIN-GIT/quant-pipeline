"""Parameter-trial enumeration, registry and deterministic selection (spec 11.3, 11.4)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc

from .config import Config
from .decision import TrialParams
from .hashing import hash_json_obj

TRIAL_DIMENSIONS = {
    "agents.defensive.overlay_gamma_candidates": lambda c: c.agents.defensive.overlay_gamma_candidates,
    "agents.mean_reversion.entry_z_candidates": lambda c: c.agents.mean_reversion.entry_z_candidates,
    "agents.trend.horizon_weight_candidates": lambda c: c.agents.trend.horizon_weight_candidates,
    "objective.lambda_cost_candidates": lambda c: c.objective.lambda_cost_candidates,
    "objective.lambda_risk_candidates": lambda c: c.objective.lambda_risk_candidates,
    "objective.lambda_track_candidates": lambda c: c.objective.lambda_track_candidates,
}


class SamplerFailure(RuntimeError):
    """The required number of distinct configurations was not reached within the attempted-point bound."""


@dataclass(frozen=True)
class TrialDesign:
    dimension_order: tuple[str, ...]
    sizes: tuple[int, ...]
    cardinality: int
    target: int
    max_attempted_points: int
    stop_reason: str
    points: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]  # distinct configurations in Sobol-index order
    seed: int
    design_hash: str


def _dedupe(values: list[Any]) -> list[Any]:
    seen, out = set(), []
    for v in values:
        k = json.dumps(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def trial_hash(params: dict[str, Any]) -> str:
    return hash_json_obj(params)


def enumerate_trials(cfg: Config) -> TrialDesign:
    """Scrambled Sobol over non-singleton dimensions sorted by canonical name; coordinates map by ``floor(u*n)``."""
    cands = {name: _dedupe(list(fn(cfg))) for name, fn in TRIAL_DIMENSIONS.items()}
    dims = sorted(n for n, v in cands.items() if len(v) > 1)
    sizes = tuple(len(cands[n]) for n in dims)
    cardinality = math.prod(sizes) if dims else 1
    cap = cfg.evaluation.max_parameter_trials
    target = min(cap, cardinality)
    max_points = max(1024, 32 * cardinality)
    seed = cfg.reproducibility.random_streams.trial_sobol
    singles = {n: v[0] for n, v in cands.items() if len(v) == 1}
    points, trials, seen = [], [], {}
    engine = qmc.Sobol(d=max(len(dims), 1), scramble=True, seed=seed)
    for idx in range(max_points):
        u = engine.random(1)[0]
        mapped = [min(math.floor(u[j] * sizes[j]), sizes[j] - 1) for j in range(len(dims))]
        params = {**singles, **{n: cands[n][m] for n, m in zip(dims, mapped, strict=True)}}
        h = trial_hash(params)
        dup = h in seen
        points.append({"point_index": idx, "u": [float(x) for x in u[: len(dims)]], "mapped": mapped, "trial_hash": h,
                       "duplicate_of_point": seen.get(h), "counts_toward_cap": not dup})
        if not dup:
            seen[h] = idx
            trials.append({"trial_hash": h, "point_index": idx, "parameters": params})
        if len(seen) >= target:
            break
    else:
        raise SamplerFailure(f"only {len(seen)} of {target} distinct configurations in {max_points} points")
    stop = "candidate_space_exhausted" if target == cardinality and cardinality <= cap else "cap_reached"
    if cardinality == cap:
        stop = "cap_reached"
    body = {"dimension_order": dims, "sizes": list(sizes), "seed": seed, "points": points, "target": target}
    return TrialDesign(tuple(dims), sizes, cardinality, target, max_points, stop, tuple(points), tuple(trials), seed, hash_json_obj(body))


def to_trial_params(cfg: Config, parameters: dict[str, Any], gate_name: str | None = None) -> TrialParams:
    return TrialParams(parameters["agents.defensive.overlay_gamma_candidates"], parameters["agents.mean_reversion.entry_z_candidates"],
                       tuple(parameters["agents.trend.horizon_weight_candidates"]), gate_name or cfg.regime.agent_gate_candidates[0].name,
                       parameters["objective.lambda_track_candidates"], parameters["objective.lambda_risk_candidates"],
                       parameters["objective.lambda_cost_candidates"])


@dataclass
class TrialRegistry:
    """Append-only registry keyed by ``(run_id, fold_id, trial_hash)``; every distinct hash counts toward the cap."""

    run_id: str
    cap: int
    rows: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)

    def register(self, fold_id: str, trial_hash_: str, status: str, **info: Any) -> None:
        key = (self.run_id, fold_id, trial_hash_)
        if key in self.rows:
            raise ValueError(f"registry is append-only; {key} already recorded")
        if len({k[2] for k in self.rows if k[1] == fold_id}) >= self.cap:
            raise RuntimeError("trial cap breach")
        self.rows[key] = {"status": status, **info}


@dataclass(frozen=True)
class Selection:
    winner: str
    ranking: tuple[str, ...]
    tie_path: tuple[str, ...]
    margin: float
    mean_scores: dict[str, float]
    stable: bool


def select(scores: dict[str, list[float]], turnover: dict[str, float], complexity: dict[str, float], tolerance: float,
           eligibility: dict[str, bool] | None = None) -> Selection:
    """Mean inner score; ties (within ``tolerance``) break on lower turnover, lower complexity, then smaller trial hash."""
    elig = eligibility or dict.fromkeys(scores, True)
    mean = {h: float(np.mean(s)) for h, s in scores.items() if elig.get(h, True) and len(s) and np.all(np.isfinite(s))}
    if not mean:
        raise ValueError("no eligible trial")
    best = max(mean.values())
    tied = sorted((h for h, m in mean.items() if best - m <= tolerance), key=lambda h: (turnover[h], complexity[h], h))
    path = ["score"]
    if len(tied) > 1:
        path.append("turnover")
        if turnover[tied[0]] == turnover[tied[1]]:
            path.append("complexity")
            if complexity[tied[0]] == complexity[tied[1]]:
                path.append("trial_hash")
    rest = sorted((h for h in mean if h not in tied), key=lambda h: (-mean[h], h))
    ranking = tuple(tied + rest)
    second = mean[ranking[1]] if len(ranking) > 1 else float("nan")
    return Selection(ranking[0], ranking, tuple(path), float(mean[ranking[0]] - second) if len(ranking) > 1 else float("nan"), mean, True)
