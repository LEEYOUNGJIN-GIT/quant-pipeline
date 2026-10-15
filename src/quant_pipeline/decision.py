"""Decision assembly (spec 8): regime -> agents -> gates -> mixture -> Defensive envelope, and the train-only virtual
pretrade path that calibrates the objective (spec 9.2-9.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from . import agents as A
from .aggregation import Combined, combine
from .calendars import Calendars, utc
from .config import Config
from .covariance import fit_ledoit_wolf
from .data import MarketData
from .fitstate import FoldFit, agent_context
from .objective import ObjectiveCalibration, calibrate_objective, drift
from .rates import RateCurve, act_days
from .regimes import FixedRegimeTable
from .risk_pressure import Threshold, calibrate, metric_series, pressure

RUNNERS = {"Trend": A.trend.run, "MeanReversion": A.mean_reversion.run, "Defensive": A.defensive.run,
           "Diversification": A.diversification.run}


class RegimeSource(Protocol):
    def probs(self, cutoff: pd.Timestamp) -> tuple[np.ndarray, list[str]]: ...


@dataclass(frozen=True)
class FixedRegimeSource:
    """Fixture decision lane: the fixed regime table looked up as-of the decision cutoff (spec 7.5)."""

    table: FixedRegimeTable
    labels: tuple[str, ...] = ("low_risk", "high_risk")

    def probs(self, cutoff: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
        return self.table.asof(cutoff)[0], list(self.labels)


@dataclass(frozen=True)
class TrialParams:
    """Candidate parameters that shape targets. A trial overlay changes these, never the base profile."""

    overlay_gamma: float
    entry_z: float
    horizon_weights: tuple[float, ...]
    gate_name: str
    lambda_track: float
    lambda_risk: float
    lambda_cost: float
    n_components: int | None = None  # research trial dimension (OQ-043); None = the profile's first candidate (the fixture's only one)

    @classmethod
    def first(cls, cfg: Config, gate_name: str | None = None) -> TrialParams:
        o = cfg.objective
        return cls(cfg.agents.defensive.overlay_gamma_candidates[0], cfg.agents.mean_reversion.entry_z_candidates[0],
                   tuple(cfg.agents.trend.horizon_weight_candidates[0]), gate_name or cfg.regime.agent_gate_candidates[0].name,
                   o.lambda_track_candidates[0], o.lambda_risk_candidates[0], o.lambda_cost_candidates[0])


@dataclass(frozen=True)
class DecisionTarget:
    cutoff: pd.Timestamp
    combined: Combined
    outputs: tuple[A.AgentOutput, ...]
    thresholds: dict[str, Threshold]
    pressure_components: dict[str, float]
    regime_probabilities: tuple[float, ...]
    a_bar: np.ndarray  # risky assets then CASH_KRW, canonical order


def pressure_components(cfg: Config, ctx: A.AgentContext) -> tuple[dict[str, float], dict[str, Threshold]]:
    """Observed pressure at the last row against thresholds fitted from this context's training rows only."""
    rp = cfg.risk_pressure_calibration
    ill = ctx.illiquid_history if ctx.illiquid_history is not None else np.zeros(len(ctx.returns))
    series = metric_series(ctx.returns, ill, drawdown_window=cfg.features.drawdown_window, factor_window=rp.factor_window_rows,
                           factor_min_obs=rp.factor_min_observations, first_row=ctx.train_start_row)
    last = len(ctx.returns) - 1
    names = (("market_drawdown_magnitude", "drawdown"), ("first_pca_variance_share", "factor"), ("illiquid_fraction", "liquidity"))
    comps, ths = {}, {}
    for metric, key in names:
        th = calibrate(metric, series[metric][ctx.train_start_row: ctx.train_end_row + 1], start_q=rp.start_quantile,
                       full_q=rp.full_quantile, gap_floor=rp.threshold_gap_floor, min_decisions=rp.minimum_training_decisions,
                       fallback=tuple(rp.fallback[key]))
        x = float(series[metric][last])
        comps[key] = pressure(x, th.start, th.full, rp.threshold_gap_floor) if np.isfinite(x) else 0.0
        ths[metric] = th
    return comps, ths


def decide(cfg: Config, ctx: A.AgentContext, regime: RegimeSource, trial: TrialParams, cutoff: pd.Timestamp,
           memo: dict | None = None) -> DecisionTarget:
    """Unoptimized aggregate target ``a_bar`` for one context (spec 8.3)."""
    kw = {"Trend": {"horizon_weights": list(trial.horizon_weights)}, "MeanReversion": {"entry_z": trial.entry_z}}
    outputs = []
    for k in A.AGENTS:
        if memo is not None and k == "Diversification":
            # Diversification reads only the fold's covariance, the eligible set and the caps: no trial dimension can change it,
            # so one solve per (fit, cutoff) serves every gamma / lambda candidate (spec 3.4 allows sharing on exact equality)
            key = ("Diversification", ctx.fold_id, ctx.refit_id, ctx.as_of.isoformat(), ctx.covariance.tobytes(),
                   ctx.tradable.tobytes(), ctx.max_weights.tobytes())
            if key not in memo:
                memo[key] = RUNNERS[k](ctx, cfg, **kw.get(k, {}))
            outputs.append(memo[key])
        else:
            outputs.append(RUNNERS[k](ctx, cfg, **kw.get(k, {})))
    probs, labels = regime.probs(cutoff)
    gate = next(g for g in cfg.regime.agent_gate_candidates if g.name == trial.gate_name)
    comps, ths = pressure_components(cfg, ctx)
    d = cfg.agents.defensive
    c = combine(outputs, ctx.assets, probs, labels, gate.by_state_label, p_high=float(probs[-1]), pressure_components=comps,
                base=d.overlay_base_risky_budget, minimum=d.overlay_min_risky_budget, gamma=trial.overlay_gamma,
                target_cash_floor=cfg.constraints.target_cash_floor, max_weights=ctx.max_weights,
                usd_mask=np.array([a.currency == "USD" for a in cfg.universe.assets]),
                usd_limit=cfg.constraints.usd_exposure_limit, all_zero_fallback=cfg.regime.all_zero_gate_fallback or "equal")
    a_bar = np.array([c.a_bar[a] for a in ctx.assets] + [c.a_bar[A.CASH]])
    return DecisionTarget(utc(cutoff), c, tuple(outputs), ths, comps, tuple(float(x) for x in probs), a_bar)


def calibrate_fold(cfg: Config, md: MarketData, cals: Calendars, sess: dict[str, pd.DatetimeIndex], fit: FoldFit,
                   regime: RegimeSource, trial: TrialParams, calibration_cutoffs: list[pd.Timestamp],
                   initial_weights: np.ndarray, memo: dict | None = None) -> ObjectiveCalibration:
    """Train-only calibration (spec 9.2-9.4). Every input at calibration decision ``u`` is causal at ``u``.

    The virtual path is costless, ignores lots, starts from the marked initial holdings and drifts post-trade
    ``a_bar`` with research (total-return) returns and cash accrual. It never becomes performance or a ledger.
    """
    if any(c > fit.fit_end_utc for c in calibration_cutoffs):
        raise ValueError("calibration decisions must lie inside the permitted training window")
    curve = RateCurve.from_frame(md.rates, cfg.availability.max_rate_stale_business_days)
    a_rows, v_rows, sigmas = [], [initial_weights.copy()], []
    prev_cutoff = None
    for j, u in enumerate(calibration_cutoffs):
        ctx = agent_context(cfg, md, cals, sess, fit, u, calibration=True)
        target = decide(cfg, ctx, regime, trial, u, memo)
        a_rows.append(target.a_bar)
        window = ctx.returns[-cfg.covariance.lookback_rows:]
        if len(window) >= cfg.covariance.min_observations:
            sigmas.append(fit_ledoit_wolf(window, ctx.assets, fold_id=fit.fold_id, refit_id=fit.refit_id, fit_start_utc=u.to_pydatetime(),
                                          fit_end_utc=u.to_pydatetime(), effective_observations={},
                                          min_observations=cfg.covariance.min_observations, eigen_floor=cfg.covariance.psd_eigen_floor,
                                          max_frobenius=cfg.covariance.max_psd_repair_frobenius).matrix)
        else:
            sigmas.append(None)
        if j > 0:  # v_minus(u_j) = drift(a_bar(u_{j-1})) over the realized research returns of the new row
            days = act_days(prev_cutoff, u)
            cash_ret = float(curve.rate_in_force(prev_cutoff)) * days / 365.0
            v_rows.append(drift(a_rows[j - 1], ctx.returns[-1], cash_ret))
        prev_cutoff = u
    return calibrate_objective(cfg, np.array(a_rows), np.array(v_rows), sigmas)
