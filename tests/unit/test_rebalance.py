"""Rebalance plan (spec 6.3, 6.4, 9.5): sizing, lots, explicit shortfall, timing, hashing and immutability."""
from __future__ import annotations

import dataclasses
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.accounting import CostRates, Fill, Ledger
from quant_pipeline.calendars import Calendars
from quant_pipeline.config import load_config
from quant_pipeline.data import load_market_data
from quant_pipeline.rebalance import LedgerSnapshot, build_plan, write_plan

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
MD = load_market_data(CFG, ROOT)
CALS = Calendars.load(["XKRX", "XNYS"])
CUT = pd.Timestamp("2024-04-03T21:15:00Z")  # decision at eligible position 41 (all assets tradable)
CUT_ILLIQ = pd.Timestamp("2024-03-14T20:15:00Z")  # NYSE_C flagged liquidity_invalid
ASSETS = ("KRX_A", "KRX_B", "NYSE_C")
CASH0 = Decimal("1000000000.00")


def snap(cash=CASH0, as_of="2024-04-03T21:15:00Z", **holdings):
    return LedgerSnapshot(cash, {a: holdings.get(a, 0) for a in ASSETS}, pd.Timestamp(as_of))


def plan(target, snapshot=None, cfg=CFG, cutoff=CUT, **kw):
    args = {"run_id": "r", "fold_id": "f", "refit_id": "x", "decision_id": "D1", "trial_hash": "sha256:t"} | kw
    return build_plan(cfg, MD, CALS, snapshot or snap(), np.array(target, dtype=float), decision_cutoff=cutoff, **args)


def unit(asset, cutoff=CUT):
    p = MD.raw_mark(asset, cutoff)
    return p * (MD.fx_mark(cutoff) if asset == "NYSE_C" else 1)


def test_quantities_are_the_target_value_over_raw_price_rounded_toward_zero():
    p = plan([0.2, 0.3, 0.1, 0.4])
    nav = Decimal(p.nav_pre_cost_krw)
    for o in p.orders:
        w = Decimal(repr(float(p.target_weights[o.asset_id])))
        want = int((w * nav / unit(o.asset_id)).to_integral_value(rounding=ROUND_DOWN))
        assert o.intended_quantity == want and o.side == "buy" and o.status == "planned" and o.shortfall_quantity == 0
        assert Decimal(o.intended_quantity) * unit(o.asset_id) <= w * nav  # never rounds up
    assert p.nav_pre_cost_krw == p.cash_accrued_krw  # all-cash start: NAV is the accrued cash


def test_cash_is_accrued_to_the_decision_before_sizing():
    p = plan([0.2, 0.3, 0.1, 0.4], snap(as_of="2024-01-02T21:15:00Z"))
    assert Decimal(p.cash_accrued_krw) > CASH0  # three months of positive KRW rate since the opening holdings
    assert Decimal(p.nav_pre_cost_krw) == Decimal(p.cash_accrued_krw)
    assert plan([0.2, 0.3, 0.1, 0.4]).cash_accrued_krw == str(CASH0)  # snapshot already at the decision: nothing accrues


def test_lot_sizes_greater_than_one_round_toward_zero_to_a_lot_multiple():
    assets = [a.model_copy(update={"lot_size": 100}) for a in CFG.universe.assets]
    cfg = CFG.model_copy(update={"universe": CFG.universe.model_copy(update={"assets": assets})})
    for o in plan([0.2, 0.3, 0.1, 0.4], cfg=cfg).orders:
        assert o.intended_quantity % 100 == 0 and o.lot_size == 100


def test_sells_and_buys_from_existing_holdings():
    p = plan([0.0, 0.5, 0.0, 0.5], snap(cash=Decimal("1000.00"), KRX_A=1000, KRX_B=100, NYSE_C=50))
    by = {o.asset_id: o for o in p.orders}
    assert by["KRX_A"].side == "sell" and by["KRX_A"].intended_trade_quantity == -1000 and by["KRX_A"].intended_quantity == 0
    assert by["NYSE_C"].side == "sell" and by["NYSE_C"].intended_trade_quantity == -50
    assert by["KRX_B"].pretrade_quantity == 100


def test_cash_feasibility_reduces_the_lowest_priority_buy_and_records_explicit_shortfall():
    p = plan([0.4, 0.3, 0.2999, 0.0001])  # buys consume ~all NAV; costs push projected cash below zero
    assert Decimal(p.projected_post_cost_cash_krw) >= 0
    reduced = [o for o in p.orders if o.shortfall_quantity]
    assert reduced and all(o.shortfall_reason == "cash_feasibility" for o in reduced)
    assert p.summary["shortfall_orders"] == [o.asset_id for o in reduced]
    # the smallest target-dollar buy has the lowest priority and is the one reduced first
    values = {o.asset_id: abs(o.unconstrained_trade_quantity) * unit(o.asset_id) for o in p.orders}
    assert reduced[0].asset_id == min(values, key=values.get)
    # minimal reduction: giving one more lot back to the reduced order would make projected cash negative
    r = reduced[0]
    usd = r.asset_id == "NYSE_C"
    bps = Decimal(5 + 5 + (2 if usd else 0)) / 10000
    one_lot_total = unit(r.asset_id) * (1 + bps)
    assert Decimal(p.projected_post_cost_cash_krw) < one_lot_total + Decimal(1)
    assert r.intended_trade_quantity > 0 and r.shortfall_quantity == r.unconstrained_trade_quantity - r.intended_trade_quantity


def test_nothing_is_ordered_for_an_untradable_asset_and_its_quantity_is_kept():
    p = plan([0.2, 0.2, 0.4, 0.2], snap(NYSE_C=100), cutoff=CUT_ILLIQ)
    assert "NYSE_C" not in {o.asset_id for o in p.orders} and p.summary["untradable"] == ["NYSE_C"]


def test_no_trade_plan_is_explicit():
    p = plan([0.0, 0.0, 0.0, 1.0])
    assert p.orders == () and p.no_trade_reason == "no_trade_required" and p.plan_content_hash.startswith("sha256:")


def test_execution_window_is_strictly_after_the_decision_and_bounded_by_order_age():
    p = plan([0.2, 0.3, 0.1, 0.4])
    for o in p.orders:
        ex = next(a.exchange for a in CFG.universe.assets if a.asset_id == o.asset_id)
        first = pd.Timestamp(o.intended_execution_utc)
        assert first > CUT and first == CALS.next_close_after(ex, CUT)[1]
        assert pd.Timestamp(o.max_execution_utc) > first  # two further sessions of tolerance (max_order_age_sessions = 2)
    assert {o.intended_execution_utc for o in p.orders if o.asset_id.startswith("KRX")} == {"2024-04-04T06:30:00Z"}


def test_projected_costs_and_cash_equal_what_the_ledger_actually_posts():
    s = snap(cash=Decimal("5000000.00"), KRX_A=20, NYSE_C=10)
    p = plan([0.2, 0.2, 0.2, 0.4], s)
    led = Ledger(cash=Decimal(p.cash_accrued_krw), holdings=dict(s.holdings))
    total = Decimal(0)
    for o in sorted(p.orders, key=lambda o: (o.side != "sell", o.asset_id)):  # sells post before buys
        usd = o.asset_id == "NYSE_C"
        rates = CostRates(Decimal(5), Decimal(5), Decimal(0), Decimal(2) if usd else Decimal(0))
        fx = Decimal(o.sizing_fx) if usd else None
        res = led.post_fill(Fill(o.asset_id, "USD" if usd else "KRW", o.intended_trade_quantity, Decimal(o.sizing_price_raw), fx), rates)
        assert res["cost"] == Decimal(o.expected_cost_krw)
        total += res["cost"]
    assert led.cash == Decimal(p.projected_post_cost_cash_krw) and total == sum(Decimal(o.expected_cost_krw) for o in p.orders)


def test_cost_multiplier_scales_projected_costs():
    c0 = sum(Decimal(o.expected_cost_krw) for o in plan([0.2, 0.3, 0.1, 0.4], multiplier=Decimal(0)).orders)
    c1 = sum(Decimal(o.expected_cost_krw) for o in plan([0.2, 0.3, 0.1, 0.4], multiplier=Decimal(1)).orders)
    c2 = sum(Decimal(o.expected_cost_krw) for o in plan([0.2, 0.3, 0.1, 0.4], multiplier=Decimal(2)).orders)
    assert c0 == 0 and c1 > 0 and abs(c2 - 2 * c1) <= Decimal("0.05")


def test_plan_hash_is_stable_and_sensitive_to_every_input():
    base = plan([0.2, 0.3, 0.1, 0.4])
    assert plan([0.2, 0.3, 0.1, 0.4]).plan_content_hash == base.plan_content_hash
    assert plan([0.2, 0.3, 0.1000001, 0.3999999]).plan_content_hash != base.plan_content_hash
    assert plan([0.2, 0.3, 0.1, 0.4], trial_hash="sha256:other").plan_content_hash != base.plan_content_hash
    assert plan([0.2, 0.3, 0.1, 0.4], decision_id="D2").plan_content_hash != base.plan_content_hash
    assert plan([0.2, 0.3, 0.1, 0.4], snap(KRX_A=1)).plan_content_hash != base.plan_content_hash
    assert plan([0.2, 0.3, 0.1, 0.4], cutoff=pd.Timestamp("2024-04-04T21:15:00Z")).plan_content_hash != base.plan_content_hash


def test_a_plan_is_frozen_and_never_rewritten(tmp_path):
    p = plan([0.2, 0.3, 0.1, 0.4])
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.nav_pre_cost_krw = "1"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.orders[0].intended_quantity = 0  # type: ignore[misc]
    manifest = write_plan(p, tmp_path)
    assert manifest.exists() and pd.read_parquet(tmp_path / "rebalance_plan.parquet").plan_content_hash.eq(p.plan_content_hash).all()
    with pytest.raises(FileExistsError):
        write_plan(p, tmp_path)  # immutable
    assert write_plan(plan([0.0, 0.0, 0.0, 1.0]), tmp_path / "empty").exists()


def test_the_plan_records_origin_ids_and_every_spec_6_3_field():
    f = plan([0.2, 0.3, 0.1, 0.4]).frame()
    need = {"run_id", "fold_id", "refit_id", "decision_id", "order_id", "asset_id", "side", "target_absolute_weight",
            "pretrade_quantity", "intended_quantity", "intended_trade_quantity", "lot_size", "sizing_price_raw", "sizing_fx",
            "decision_cutoff_utc", "intended_execution_utc", "max_execution_utc", "status", "plan_content_hash", "trial_hash"}
    assert need <= set(f.columns) and f.order_id.is_unique and set(f.fold_id) == {"f"}
    assert f[f.asset_id == "NYSE_C"].sizing_fx.iloc[0] != "" and f[f.asset_id == "KRX_A"].sizing_fx.iloc[0] == ""
