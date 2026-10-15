"""Interval and exclusion report (INV-OOS-003): market-specific endpoints, excluded rows with causes, no duplicate scoring."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from quant_pipeline.calendars import Calendars, build_row_map
from quant_pipeline.config import load_config
from quant_pipeline.intervals import interval_report
from quant_pipeline.splits import build_split_manifest

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
CALS = Calendars.load(["XKRX", "XNYS"])
RM, _ = build_row_map(CFG, CALS)
M = build_split_manifest(CFG, CALS, RM)
R = interval_report(CFG, CALS, RM, M)


def block(name):
    return next(b for b in R["blocks"] if b["block"] == name)


def test_every_block_has_market_specific_endpoints_and_a_null_label_with_a_reason():
    assert len(R["blocks"]) == 6 + 2 + 1  # three passes x two validation blocks, two outer tests, one holdout
    for b in R["blocks"]:
        assert set(b["market_endpoints"]) == {"XKRX", "XNYS"} and b["label_interval"]["value"] is None and b["label_interval"]["reason"]
        for e in b["market_endpoints"].values():
            assert e["first_execution_close_utc"] < e["last_permitted_execution_close_utc"] and e["duration_hours"] > 0
        assert b["market_endpoints"]["XKRX"]["last_permitted_execution_close_utc"] < b["market_endpoints"]["XNYS"]["last_permitted_execution_close_utc"] \
            or b["market_endpoints"]["XKRX"]["first_execution_close_utc"] < b["market_endpoints"]["XNYS"]["first_execution_close_utc"]  # markets differ


def test_excluded_rows_and_causes_follow_the_declared_geometry():
    b = block("outer-1-inner-1")
    assert b["decision_positions"] == [30, 34] and b["censored_decision"]["id"] == "D034"
    assert {(x["id"], x["cause"]) for x in b["excluded_observations"]} == {
        ("D028", "purge_before_validation"), ("D029", "purge_before_validation"), ("D035", "gap_after_validation")}
    assert block("outer-1-inner-2")["excluded_observations"][0]["cause"] == "purge_before_validation"  # the last block has no gap row
    assert not any(x["cause"].startswith("gap") for x in block("outer-1-inner-2")["excluded_observations"])


def test_scored_intervals_are_counted_per_block_and_never_duplicated():
    assert [block(f"outer-{k}-test")["scored_holding"]["intervals"] for k in (1, 2)] == [11, 11]
    assert block("holdout")["scored_holding"]["intervals"] == 12
    assert R["overlap_findings"] == [] and R["duplicate_scoring"] is False


def test_the_report_detects_duplicate_scoring_and_an_excluded_row_inside_a_block():
    dup_outer = dataclasses.replace(M.outer[1], test=dataclasses.replace(M.outer[1].test, valuation_sessions=M.outer[0].test.valuation_sessions))
    bad = dataclasses.replace(M, outer=(M.outer[0], dup_outer))
    assert any("outer test blocks share" in f for f in interval_report(CFG, CALS, RM, bad)["overlap_findings"])
    leak = dataclasses.replace(M, holdout=dataclasses.replace(M.holdout, valuation_sessions=M.outer[0].test.valuation_sessions[:1]))
    assert any("also a holdout interval" in f for f in interval_report(CFG, CALS, RM, leak)["overlap_findings"])
    p = M.passes[0]
    f0 = dataclasses.replace(p.inner[0], purge_positions=(p.inner[0].validation.start_pos,))
    inside = dataclasses.replace(M, passes=(dataclasses.replace(p, inner=(f0, *p.inner[1:])), *M.passes[1:]))
    assert any("excluded row lies inside" in f for f in interval_report(CFG, CALS, RM, inside)["overlap_findings"])


@pytest.mark.parametrize("scope", ["outer_oos", "holdout"])
def test_oos_scopes_have_no_excluded_rows(scope):
    assert all(b["excluded_observations"] == [] for b in R["blocks"] if b["scope"] == scope)
