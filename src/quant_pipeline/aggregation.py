"""Agent score -> absolute cash-inclusive target (spec 8.1, 16.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agents.base import AGENTS, CASH, AgentOutput


def score_to_target(raw_scores: np.ndarray, confidence: float, risk_budget_cap: float) -> np.ndarray:
    """Positive part of raw scores is normalized over risky assets; risky gross = cap * confidence.

    Returns ``[w_1..w_n, cash]``. Negative scores are never shorted or renormalized into other
    assets; if no positive mass exists the target is exactly all cash.
    """
    s = np.asarray(raw_scores, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("raw scores must be finite")
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= risk_budget_cap <= 1.0:
        raise ValueError("confidence and risk_budget_cap must lie in [0, 1]")
    positive = np.clip(s, 0.0, None)
    mass = positive.sum()
    if mass <= 0.0:
        out = np.zeros(s.size + 1)
        out[-1] = 1.0
        return out
    budget = risk_budget_cap * confidence
    risky = budget * positive / mass
    return np.append(risky, 1.0 - risky.sum())


# ------------------------------------------------------------------------------ spec 8.3: gates, mixture, envelope
def raw_gates(p_states: np.ndarray, labels: list[str], by_state_label: dict[str, dict[str, float]]) -> np.ndarray:
    """g_k = sum_s p_s * G[label_s][agent_k], in canonical agent order. Labels are aligned (volatility ascending)."""
    p = np.asarray(p_states, dtype=float)
    if p.shape != (len(labels),) or not np.isclose(p.sum(), 1.0, atol=1e-9) or np.any(p < 0):
        raise ValueError("state probabilities must be a distribution over the labelled states")
    return np.array([sum(p[s] * by_state_label[labels[s]][k] for s in range(len(labels))) for k in AGENTS])


def normalize_gates(g: np.ndarray, abstaining: np.ndarray, fallback: str = "equal") -> np.ndarray | None:
    """Normalize over non-abstaining agents. All-zero gates use the named fallback; all abstaining returns None."""
    live = ~np.asarray(abstaining, dtype=bool)
    if not live.any():
        return None
    gm = np.where(live, g, 0.0)
    if gm.sum() <= 0.0:
        if fallback != "equal":
            raise ValueError(f"unknown all-zero gate fallback {fallback!r}")
        gm = live.astype(float)
    return gm / gm.sum()


def defensive_envelope(risk_pressure: float, base: float, minimum: float, gamma: float) -> float:
    return minimum + (base - minimum) * (1.0 - risk_pressure) ** gamma


@dataclass(frozen=True)
class Combined:
    a_bar: dict[str, float]  # risky assets + CASH_KRW, absolute weights summing to one
    b_proposed: float
    b_defensive: float
    b_final: float
    gates_raw: tuple[float, ...]
    gates_normalized: tuple[float, ...] | None
    abstaining: tuple[str, ...]
    risk_pressure: float
    q_bar: tuple[float, ...]
    diagnostics: dict[str, Any]


def apply_caps(weights: np.ndarray, max_weights: np.ndarray, usd_mask: np.ndarray, usd_limit: float) -> np.ndarray:
    """Per-asset and currency caps only ever reduce risky weight; the residual becomes cash (spec 8.3)."""
    w = np.minimum(weights, max_weights)
    usd = float(w[usd_mask].sum())
    if usd > usd_limit and usd > 0:
        w = w.copy()
        w[usd_mask] *= usd_limit / usd
    return w


def combine(outputs: list[AgentOutput], assets: tuple[str, ...], p_states: np.ndarray, labels: list[str],
            by_state_label: dict[str, dict[str, float]], *, p_high: float, pressure_components: dict[str, float],
            base: float, minimum: float, gamma: float, target_cash_floor: float, max_weights: np.ndarray,
            usd_mask: np.ndarray, usd_limit: float, all_zero_fallback: str = "equal") -> Combined:
    by_name = {o.agent_name: o for o in outputs}
    if set(by_name) != set(AGENTS):
        raise ValueError("combine needs exactly the four agent outputs")
    ordered = [by_name[k] for k in AGENTS]
    abstain = np.array([o.diagnostics.get("agent_status") == "degraded_no_meaningful_budget" for o in ordered])
    g = raw_gates(p_states, labels, by_state_label)
    ghat = normalize_gates(g, abstain, all_zero_fallback)
    pressure = max(p_high, *pressure_components.values()) if pressure_components else p_high
    b_def = defensive_envelope(pressure, base, minimum, gamma)
    n = len(assets)
    if ghat is None:
        b_prop, q_bar = 0.0, np.zeros(n)
    else:
        b_prop = float(sum(ghat[i] * ordered[i].proposed_risky_budget for i in range(len(AGENTS))))
        mix = sum(ghat[i] * np.array([ordered[i].normalized_score[a] for a in assets]) for i in range(len(AGENTS)))
        q_bar = mix / mix.sum() if mix.sum() > 0 else np.zeros(n)
    b_final = 0.0 if q_bar.sum() <= 0 else float(min(b_prop, b_def, 1.0 - target_cash_floor))
    w = apply_caps(b_final * q_bar, max_weights, usd_mask, usd_limit)
    a_bar = {a: float(x) for a, x in zip(assets, w, strict=True)}
    a_bar[CASH] = float(1.0 - w.sum())
    return Combined(a_bar, b_prop, b_def, b_final, tuple(g), None if ghat is None else tuple(ghat),
                    tuple(o.agent_name for o, ab in zip(ordered, abstain, strict=True) if ab), pressure, tuple(q_bar),
                    {"pressure_components": {"p_high": p_high, **pressure_components},
                     "cap_reduction": float(b_final - w.sum())})
