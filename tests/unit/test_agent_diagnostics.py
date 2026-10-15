"""Pairwise agent diagnostics (INV-AGENT-004): defined values, explicit nulls, and detection of an alias."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_pipeline.agents import AGENTS, CASH, AgentOutput
from quant_pipeline.diagnostics import max_flagged_pairs, pairwise

ASSETS = ("A", "B", "C")


def out(name, w, budget=None):
    risky = dict(zip(ASSETS, w, strict=True))
    return AgentOutput("f", "r", datetime(2024, 1, 1, tzinfo=UTC), name, {}, {}, 1.0, sum(w) if budget is None else budget,
                       {**risky, CASH: 1 - sum(w)}, 1.0)


def decisions(weights_by_agent):
    n = len(next(iter(weights_by_agent.values())))
    return [{a: out(a, weights_by_agent[a][i]) for a in AGENTS} for i in range(n)]


VARY = [(0.1 + 0.05 * i, 0.2, 0.3 - 0.02 * i) for i in range(6)]


def test_identical_agents_have_unit_cosine_and_unit_correlation():
    r = pairwise(decisions(dict.fromkeys(AGENTS, VARY)), ASSETS)
    pair = r["pairs"]["Defensive|Diversification"]
    assert r["valid_simultaneous_decisions"] == 6 and pair["risky_share_cosine"]["median"] == pytest.approx(1.0)
    assert pair["risky_share_cosine"]["fraction_above_0.99"] == 1.0 and pair["asset_target_correlation"]["A"]["value"] == pytest.approx(1.0)
    flagged = max_flagged_pairs({"scopes": {"s": r}})["s"]
    assert len(flagged) == 6  # every pair is an alias


def test_different_agents_are_not_flagged_and_anti_correlated_assets_are_negative():
    other = [(0.3 - 0.05 * i, 0.2, 0.1 + 0.02 * i) for i in range(6)]
    r = pairwise(decisions({**dict.fromkeys(AGENTS, VARY), "Diversification": other}), ASSETS)
    pair = r["pairs"]["Defensive|Diversification"]
    assert pair["asset_target_correlation"]["A"]["value"] == pytest.approx(-1.0)
    assert "Defensive|Diversification" not in max_flagged_pairs({"scopes": {"s": r}})["s"]


def test_constant_series_and_short_scopes_are_null_with_a_reason_never_zero():
    r = pairwise(decisions(dict.fromkeys(AGENTS, VARY)), ASSETS)["pairs"]["Trend|MeanReversion"]
    assert r["asset_target_correlation"]["B"] == {"value": None, "reason": "a constant series has no defined correlation"}
    short = pairwise(decisions(dict.fromkeys(AGENTS, VARY[:2])), ASSETS)["pairs"]["Trend|MeanReversion"]
    assert short["asset_target_correlation"]["A"]["value"] is None and "fewer than 3" in short["asset_target_correlation"]["A"]["reason"]


def test_all_cash_decisions_have_no_defined_cosine():
    r = pairwise(decisions(dict.fromkeys(AGENTS, [(0.0, 0.0, 0.0)] * 4)), ASSETS)["pairs"]["Trend|Defensive"]["risky_share_cosine"]
    assert r["count"] == 0 and r["median"] is None and r["reason"]


def test_a_decision_missing_an_agent_is_not_counted_as_simultaneous():
    ds = decisions(dict.fromkeys(AGENTS, VARY))
    del ds[0]["Trend"]
    r = pairwise(ds, ASSETS)
    assert r["valid_simultaneous_decisions"] == 5 and r["decisions_offered"] == 6
