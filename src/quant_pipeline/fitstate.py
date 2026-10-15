"""Fold/refit-local core fit (spec 7.1): panel -> standardizer -> PCA -> Ledoit-Wolf -> asynchrony diagnostics.

There is no global fit. The fold panel is rebuilt from observations whose ``available_at_utc`` is at or before the fit
cutoff, so data after the cutoff cannot influence any artifact returned here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import SPEC_VERSION, asynchrony
from .agents.base import AgentContext
from .calendars import Calendars, fmt, parse_delay, research_warmup_rows, utc
from .config import Config
from .covariance import CovarianceResult, fit_ledoit_wolf
from .data import MarketData
from .dependency import build_fingerprint, cache_key, code_hash, default_numerical_options
from .factors import PcaFit, fit_pca
from .hashing import hash_json_bytes, hash_json_obj, hash_text
from .panels import daily_panel, weekly_panel

ALGORITHM_VERSION = "core-fit-1"


@dataclass(frozen=True)
class FoldFit:
    fold_id: str
    refit_id: str
    fit_start_utc: pd.Timestamp
    fit_end_utc: pd.Timestamp
    asset_ids: tuple[str, ...]
    panel: pd.DataFrame  # daily rows with cutoff <= fit_end
    pca: PcaFit
    covariance: CovarianceResult
    asynchrony: dict[str, Any]
    fingerprint: dict[str, Any]
    key: str


def lineage_hashes(cfg: Config, root: Path) -> dict[str, str]:
    spec = root / "IMPLEMENTATION_SPEC.md"
    return {
        "spec_hash": hash_text(spec.read_bytes()) if spec.exists() else "unavailable:spec_missing",
        "data_manifest_hash": hash_json_bytes((root / cfg.data.manifest).read_bytes()),
        "base_profile_hash": hash_json_obj(cfg.model_dump(mode="json")),
        "code_hash": code_hash(Path(__file__).parent),
    }


def _relevant_parameters(cfg: Config, n_components: int) -> dict[str, Any]:
    return {
        "features": {"clip_z": cfg.features.clip_z, "min_observations": cfg.features.min_observations},
        "pca": {"n_components": n_components, "standardize": cfg.pca.standardize, "sign_rule": cfg.pca.sign_rule},
        "covariance": {k: v for k, v in cfg.covariance.model_dump(mode="json").items() if k != "diagnostics"},
        "diagnostics": cfg.covariance.diagnostics.model_dump(mode="json"),
        "panel": {"alignment": cfg.panel.alignment, "max_stale_sessions": cfg.panel.max_stale_sessions,
                  "minimum_weekly_rows": cfg.panel.minimum_weekly_rows},
        "availability": {"price_publication_delay": cfg.availability.price_publication_delay},
    }


def fit_core(cfg: Config, md: MarketData, cals: Calendars, sess: dict[str, pd.DatetimeIndex], *, fold_id: str,
             refit_id: str, fit_end_utc: pd.Timestamp, hashes: dict[str, str], n_components: int | None = None) -> FoldFit:
    assets = tuple(a.asset_id for a in (cfg.universe.assets or []))
    delay = cfg.availability.price_publication_delay
    d = parse_delay(delay)
    fit_end = utc(fit_end_utc)
    n_comp = n_components or cfg.pca.n_components_candidates[0]

    # the fold-local panel only uses sessions whose cutoff is permitted by the fit end
    xnys = sess["XNYS"][[cals.close("XNYS", s) + d <= fit_end for s in sess["XNYS"]]]
    if len(xnys) == 0:
        raise ValueError("no permitted anchor rows at this fit cutoff")
    daily = daily_panel(md.total_return, cals, xnys, delay)
    cc = daily[daily.complete_case].reset_index(drop=True)
    cols = [f"{a}_return" for a in assets]
    returns = cc[cols].to_numpy(float)
    fit_start = utc(cc.cutoff_utc.iloc[0])
    last_used = utc(cc.cutoff_utc.iloc[-1])
    if last_used > fit_end:
        raise AssertionError("a row beyond the fit cutoff entered the fit")

    pca = fit_pca(returns, assets, n_comp, cfg.features.clip_z, cfg.features.min_observations)

    lb = cfg.covariance.lookback_rows
    window = cc.iloc[-lb:]
    pair = asynchrony.pairwise_counts(daily, list(assets))
    cov = fit_ledoit_wolf(window[cols].to_numpy(float), assets, fold_id=fold_id, refit_id=refit_id,
                          fit_start_utc=utc(window.cutoff_utc.iloc[0]).to_pydatetime(), fit_end_utc=last_used.to_pydatetime(),
                          effective_observations=pair, min_observations=cfg.covariance.min_observations,
                          eigen_floor=cfg.covariance.psd_eigen_floor, max_frobenius=cfg.covariance.max_psd_repair_frobenius)

    wk = weekly_panel(md.total_return, xnys[0], xnys[-1])
    wk = wk[pd.to_datetime(wk.cutoff_utc, utc=True) <= fit_end].reset_index(drop=True)
    xk = sess["XKRX"][[cals.close("XKRX", s) + d <= fit_end for s in sess["XKRX"]]]
    alt = daily_panel(md.total_return, cals, xk, delay, anchor_calendar="XKRX")
    diag = asynchrony.build_diagnostics(daily, wk, alt, list(assets), list(cfg.covariance.diagnostics.lag_rows), cov.matrix,
                                        lb, cfg.panel.minimum_weekly_rows, cfg.covariance.min_observations)

    obs_hash = hash_json_obj({"cutoffs": cc.cutoff_utc.tolist(), "returns": [[repr(float(v)) for v in r] for r in returns]})
    fp = build_fingerprint(
        artifact_type="core_fit_state", schema_version=SPEC_VERSION, algorithm_component_version=ALGORITHM_VERSION,
        spec_hash=hashes["spec_hash"], code_hash_=hashes["code_hash"], lock_hash=None,
        data_manifest_hash=hashes["data_manifest_hash"], base_profile_hash=hashes["base_profile_hash"], asset_order=assets,
        information_cutoff_utc=fmt(fit_end), fit_start_utc=fmt(fit_start), fit_end_utc=fmt(last_used),
        observation_set_hash=obs_hash, upstream_content_hashes={}, relevant_parameter_subset=_relevant_parameters(cfg, n_comp),
        seed=None, solver_versions={"exchange_calendars": __import__("exchange_calendars").__version__,
                                    "scikit_learn": __import__("sklearn").__version__},
        numerical_options=default_numerical_options(),
        trial_influence=({"influenced": False, "proof": "core fit reads no trial-overlay dimension",
                          "excluded_dimensions": ["agents.*", "objective.*", "regime.agent_gate_candidates",
                                                  "constraints.*", "costs.*"]} if n_components is None else
                         {"influenced": True, "parameters": ["pca.n_components_candidates"],
                          "proof": "the PCA component count is a research trial dimension; no other trial dimension is read",
                          "excluded_dimensions": ["agents.*", "objective.*", "regime.agent_gate_candidates", "constraints.*", "costs.*"]}))
    return FoldFit(fold_id, refit_id, fit_start, last_used, assets, daily, pca, cov, diag, fp, cache_key(fp))


def decision_cutoff(cals: Calendars, cfg: Config, session: pd.Timestamp) -> pd.Timestamp:
    return cals.close("XNYS", session) + parse_delay(cfg.availability.price_publication_delay)


def asset_returns_matrix(panel: pd.DataFrame, assets: tuple[str, ...]) -> np.ndarray:
    return panel[panel.complete_case][[f"{a}_return" for a in assets]].to_numpy(float)


def agent_context(cfg: Config, md: MarketData, cals: Calendars, sess: dict[str, pd.DatetimeIndex], fit: FoldFit,
                  decision_cutoff_utc: pd.Timestamp, calibration: bool = False) -> AgentContext:
    """Assemble one decision's inputs: causal history through the cutoff plus the frozen fold fit (spec 5.1, 8.2).

    ``calibration=True`` builds a training-window decision: the history is cut at the decision and the training range
    used for references/thresholds ends at that same row, so nothing later than the decision is ever consulted.
    """
    assets = fit.asset_ids
    d = parse_delay(cfg.availability.price_publication_delay)
    cut = utc(decision_cutoff_utc)
    if cut < fit.fit_end_utc and not calibration:
        raise ValueError("a decision may not precede the fit it uses")
    if calibration and cut > fit.fit_end_utc:
        raise ValueError("a calibration decision must lie inside the training window")
    xnys = sess["XNYS"][[cals.close("XNYS", s) + d <= cut for s in sess["XNYS"]]]
    panel = daily_panel(md.total_return, cals, xnys, cfg.availability.price_publication_delay)
    cc = panel[panel.complete_case]
    returns = cc[[f"{a}_return" for a in assets]].to_numpy(float)
    n_fit = int((pd.to_datetime(cc.cutoff_utc, utc=True) <= fit.fit_end_utc).sum())
    std = fit.pca.standardizer
    caps = np.array([a.max_weight for a in (cfg.universe.assets or [])])
    row_times = [utc(c) for c in cc.cutoff_utc]
    illiquid = np.array([1.0 - np.mean([md.tradable(a, t) for a in assets]) for t in row_times])
    return AgentContext(
        fold_id=fit.fold_id, refit_id=fit.refit_id, as_of=cut.to_pydatetime(), assets=assets, returns=returns,
        tradable=np.array([md.tradable(a, cut) for a in assets]), loadings=fit.pca.loadings,
        standardizer_mean=std.mean, standardizer_scale=std.scale, covariance=fit.covariance.matrix, max_weights=caps,
        train_start_row=cfg.data.fixture_window.warmup_rows if cfg.data.fixture_window else research_warmup_rows(cfg), train_end_row=(len(returns) - 1) if calibration else n_fit - 1,
        illiquid_history=illiquid, row_cutoffs=tuple(cc.cutoff_utc))


def hmm_features(fit: FoldFit, returns: np.ndarray) -> np.ndarray:
    """HMM observations (OQ-015): [first PCA factor score, mean absolute standardized return] per complete-case row."""
    std = fit.pca.standardizer
    z = (returns - std.mean) / std.scale
    return np.column_stack([z @ fit.pca.loadings[0], np.abs(z).mean(axis=1)])


HMM_FEATURE_ORDER = ("pca1_score", "mean_abs_z")
