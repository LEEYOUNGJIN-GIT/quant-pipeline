"""Required asynchrony diagnostics (spec 5.4) and weekly / alternate-anchor sensitivities (spec 5.3).

The daily XNYS-anchored panel is an operational synchronization approximation: KRX and XNYS closes do not overlap, so
same-row cross-market covariance can carry lead-lag and stale-information bias. These numbers quantify that, they do not
remove it.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

WEEKLY_TO_DAILY = 5.0  # trading days per weekly return, used only to express weekly variance in daily units


def pairwise_counts(panel: pd.DataFrame, assets: list[str]) -> dict[str, int]:
    """Rows where both assets have a newly observed return (valid pairwise overlap)."""
    out = {}
    for a, b in combinations(assets, 2):
        out[f"{a}|{b}"] = int((panel[f"{a}_return"].notna() & panel[f"{b}_return"].notna()).sum())
    for a in assets:
        out[a] = int(panel[f"{a}_return"].notna().sum())
    return out


def cross_correlations(panel: pd.DataFrame, assets: list[str], lags: list[int]) -> dict[str, dict[str, float | None]]:
    """corr(r_i[t], r_j[t+lag]) in panel rows over rows where both returns exist; None if fewer than 3 pairs."""
    out: dict[str, dict[str, float | None]] = {}
    for a, b in combinations(assets, 2):
        ra, rb = panel[f"{a}_return"].to_numpy(float), panel[f"{b}_return"].to_numpy(float)
        res: dict[str, float | None] = {}
        for lag in lags:
            x, y = (ra[: len(ra) - lag], rb[lag:]) if lag >= 0 else (ra[-lag:], rb[: len(rb) + lag])
            ok = np.isfinite(x) & np.isfinite(y)
            res[str(lag)] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 and x[ok].std() > 0 and y[ok].std() > 0 else None
        out[f"{a}|{b}"] = res
    return out


def coverage_and_ages(panel: pd.DataFrame, assets: list[str]) -> dict[str, Any]:
    n = len(panel)
    per_asset: dict[str, Any] = {}
    for a in assets:
        age = panel[f"{a}_age_hours"].dropna()
        ret = panel[f"{a}_return"]
        per_asset[a] = {
            "updated_fraction": float(panel[f"{a}_updated"].mean()),
            "no_update_fraction": float(1.0 - panel[f"{a}_updated"].mean()),
            "true_zero_returns": int((ret == 0.0).sum()),
            "age_hours": {"min": float(age.min()), "median": float(age.median()), "max": float(age.max())} if len(age) else None,
        }
    return {"rows": n, "complete_case_rows": int(panel.complete_case.sum()), "row_coverage": float(panel.complete_case.mean()),
            "per_asset": per_asset}


def _spectrum(cov: NDArray[np.float64]) -> dict[str, Any]:
    eig = np.linalg.eigvalsh(cov)
    return {"eigenvalues": eig.tolist(), "leading_share": float(eig[-1] / eig.sum()),
            "condition_number": float(eig[-1] / eig[0]) if eig[0] > 0 else None}


def _corr(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


def _weights_comparison(base_cov: NDArray[np.float64], other_cov: NDArray[np.float64], assets: list[str]) -> dict[str, Any]:
    a, b = min_variance_weights(base_cov), min_variance_weights(other_cov)
    return {"portfolio": "long_only_fully_invested_minimum_variance_ridge_1e-8", "assets": list(assets),
            "daily_baseline_weights": a.tolist(), "other_weights": b.tolist(), "max_abs_weight_diff": float(np.abs(a - b).max()),
            "one_way_turnover_between_them": float(np.abs(a - b).sum() / 2)}


def _fit_cov(panel: pd.DataFrame, assets: list[str], rows: int | None) -> tuple[NDArray[np.float64], int]:
    cc = panel[panel.complete_case]
    x = cc[[f"{a}_return" for a in assets]].to_numpy(float)
    if rows is not None:
        x = x[-rows:]
    return LedoitWolf(assume_centered=False).fit(x).covariance_, int(x.shape[0])


def min_variance_weights(cov: NDArray[np.float64], ridge: float = 1e-8) -> NDArray[np.float64]:
    """Long-only fully invested minimum-variance weights with a small ridge, solved by SLSQP.

    Deliberately a different solver from the OSQP path of the Diversification agent: it is a sensitivity diagnostic, shares no code
    with the optimizer, and must not change the optimizer's solve counters. Scale-free: scaling ``cov`` by a constant leaves it unchanged.
    """
    n = cov.shape[0]
    c = cov / np.trace(cov) * n + ridge * np.eye(n)
    res = minimize(lambda q: q @ c @ q, np.full(n, 1.0 / n), jac=lambda q: 2 * c @ q, bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda q: q.sum() - 1.0, "jac": lambda q: np.ones(n)}], method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 500})
    if not res.success:
        raise ValueError(f"minimum-variance sensitivity solve failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def sensitivity(base_cov: NDArray[np.float64], other_panel: pd.DataFrame, assets: list[str], min_rows: int,
                lookback_rows: int | None, label: str, scale_to_daily: float = 1.0) -> dict[str, Any]:
    """Compare the baseline covariance with one fitted on another panel using scale-free quantities."""
    n_cc = int(other_panel.complete_case.sum())
    if n_cc < min_rows:
        return {"status": "insufficient_history", "label": label, "complete_case_rows": n_cc, "minimum_rows": min_rows}
    cov, used = _fit_cov(other_panel, assets, lookback_rows)
    w = np.full(len(assets), 1.0 / len(assets))
    base_corr, other_corr = _corr(base_cov), _corr(cov)
    return {
        "status": "ok", "label": label, "rows_used": used,
        "correlation_max_abs_diff": float(np.abs(base_corr - other_corr).max()),
        "base_spectrum": _spectrum(base_corr), "other_spectrum": _spectrum(other_corr),
        "leading_share_diff": _spectrum(other_corr)["leading_share"] - _spectrum(base_corr)["leading_share"],
        "equal_weight_daily_risk_base": float(np.sqrt(w @ base_cov @ w)),
        "equal_weight_daily_risk_other": float(np.sqrt(w @ cov @ w / scale_to_daily)),
        "optimal_weights_comparison": _weights_comparison(base_cov, cov, assets),
    }


def build_diagnostics(daily: pd.DataFrame, weekly: pd.DataFrame, alternate: pd.DataFrame, assets: list[str],
                      lag_rows: list[int], base_cov: NDArray[np.float64], lookback_rows: int, min_weekly_rows: int,
                      min_obs: int) -> dict[str, Any]:
    weekly_sens = sensitivity(base_cov, weekly, assets, min_weekly_rows, None, "weekly_friday_23:59:59Z", WEEKLY_TO_DAILY)
    if weekly_sens["status"] == "insufficient_history":
        weekly_sens["status"] = "insufficient_weekly_history"  # the status name spec 5.3 requires, with the counts kept
    return {
        "pairwise_valid_overlap": pairwise_counts(daily, assets),
        "simultaneous_update_rows": int(daily.complete_case.sum()),
        "coverage": coverage_and_ages(daily, assets),
        "cross_correlations": cross_correlations(daily, assets, lag_rows),
        "weekly_common_cutoff": weekly_sens,
        "alternate_anchor_xkrx": sensitivity(base_cov, alternate, assets, min_obs, lookback_rows, "xkrx_anchor"),
        "limitation": "daily XNYS-anchored panel is an operational synchronization approximation, not a simultaneous covariance",
    }
