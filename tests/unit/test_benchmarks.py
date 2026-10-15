"""Benchmarks and the pro-rata rule (spec 10.3, 10.3.1, 16.10). Evidence: kernel_unit on real marks; policies in fixture_path."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.benchmarks import (
    COMPARABILITY,
    LABELS,
    BenchmarkInfeasible,
    equal_weight_target,
    make_policy,
    prorata_trade,
    validate_benchmarks,
)
from quant_pipeline.calendars import Calendars
from quant_pipeline.config import load_config
from quant_pipeline.data import load_market_data
from quant_pipeline.rebalance import LedgerSnapshot, mark_state

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
MD = load_market_data(CFG, ROOT)
CALS = Calendars.load(["XKRX", "XNYS"])
CUT = pd.Timestamp("2024-04-03T21:15:00Z")
ASSETS = ("KRX_A", "KRX_B", "NYSE_C")
TGT = equal_weight_target(CFG, 0.9)


def state(cash="1000000000.00", cutoff=CUT, **hold):
    snap = LedgerSnapshot(Decimal(cash), {a: hold.get(a, 0) for a in ASSETS}, cutoff)
    return snap, mark_state(CFG, MD, CALS, snap, cutoff)


def weights_after(ms, snap, trade):
    return np.array([float(Decimal(snap.holdings[a] + trade[a]) * ms.unit[a] / ms.nav) for a in ASSETS])


def test_four_benchmarks_have_distinct_labels_and_comparability_classes():
    ids = [b.benchmark_id for b in CFG.benchmarks.definitions]
    assert ids == list(LABELS) and len(set(LABELS.values())) == 4
    assert COMPARABILITY["equal_weight_periodic"] == COMPARABILITY["equal_weight_staged_entry"] == "implementation_matched"
    assert COMPARABILITY["equal_weight_buy_and_hold_reference"] == "reference_initial_turnover_exempt"
    assert "equal_weight_buy_and_hold" not in ids  # the ambiguous single identifier is prohibited


def test_benchmark_feasibility_is_validated_before_evaluation():
    validate_benchmarks(CFG)
    few = CFG.model_copy(update={"benchmarks": CFG.benchmarks.model_copy(update={"required_benchmark_eligible_count": 4})})
    with pytest.raises(BenchmarkInfeasible, match="below the required count"):
        validate_benchmarks(few)
    small = [a.model_copy(update={"max_weight": 0.2}) for a in CFG.universe.assets]
    with pytest.raises(BenchmarkInfeasible, match="cannot cover"):
        validate_benchmarks(CFG.model_copy(update={"universe": CFG.universe.model_copy(update={"assets": small})}))
    tight = [a.model_copy(update={"max_weight": 0.25 if a.asset_id == "KRX_A" else 0.6}) for a in CFG.universe.assets]
    with pytest.raises(BenchmarkInfeasible, match="smallest member cap"):
        validate_benchmarks(CFG.model_copy(update={"universe": CFG.universe.model_copy(update={"assets": tight})}))


def test_equal_weight_target_spreads_b_over_eligible_members_and_leaves_cash():
    np.testing.assert_allclose(TGT, [0.3, 0.3, 0.3, 0.1])
    ineligible = [a.model_copy(update={"benchmark_eligible": a.asset_id != "NYSE_C"}) for a in CFG.universe.assets]
    t = equal_weight_target(CFG.model_copy(update={"universe": CFG.universe.model_copy(update={"assets": ineligible})}), 0.9)
    np.testing.assert_allclose(t, [0.45, 0.45, 0.0, 0.1])


def test_from_cash_the_turnover_limit_caps_the_first_deployment_at_L():
    snap, ms = state()
    trade, unc, reasons, info = prorata_trade(CFG, ms, snap, TGT, turnover_limit=0.5, multiplier=Decimal(1))
    assert info.required_turnover == pytest.approx(0.9) and info.scale == pytest.approx(0.5 / 0.9) and not info.exempt
    after = weights_after(ms, snap, trade)
    assert after.sum() <= 0.5 + 1e-9 and after.sum() > 0.499  # residual lots use the turnover headroom up to one lot
    assert all(q >= 0 for q in trade.values()) and all(reasons[a] == "turnover_scaling" for a in ASSETS if unc[a])
    assert all(q <= u or r for q, u, r in ((trade[a], unc[a], a in info.residual_lots_added) for a in ASSETS))


def test_the_conventional_reference_is_exempt_only_from_the_turnover_cap():
    snap, ms = state()
    trade, _, reasons, info = prorata_trade(CFG, ms, snap, TGT, turnover_limit=None, multiplier=Decimal(1))
    after = weights_after(ms, snap, trade)
    assert info.exempt and info.scale == 1.0 and after.sum() == pytest.approx(0.9, abs=2e-3)
    assert (after <= 0.6 + 1e-12).all() and after[2] <= CFG.constraints.usd_exposure_limit  # caps are not waived
    cost = sum(Decimal(0) for _ in ASSETS)  # cash solvency is not waived: projected cash stays non-negative
    from quant_pipeline.rebalance import projected_cash_after
    assert projected_cash_after(CFG, ms, trade, Decimal(1)) + cost >= 0 and not reasons


def test_quantities_round_toward_zero_and_never_exceed_the_scaled_delta_plus_residual_lots():
    snap, ms = state()
    trade, unc, _, info = prorata_trade(CFG, ms, snap, TGT, turnover_limit=0.5, multiplier=Decimal(1))
    for a in ASSETS:
        scaled = Decimal(repr(float((TGT[ASSETS.index(a)]) * info.scale))) * ms.nav
        assert Decimal(unc[a]) * ms.unit[a] <= scaled  # step 3: toward zero, never up
        assert trade[a] - unc[a] >= 0  # residual lots only ever add


def test_when_cash_is_insufficient_buys_are_reduced_pro_rata_not_borrowed():
    snap, ms = state()
    heavy = np.array([0.34, 0.33, 0.3299, 0.0001])
    trade, unc, reasons, info = prorata_trade(CFG, ms, snap, heavy, turnover_limit=None, multiplier=Decimal(1))
    from quant_pipeline.rebalance import projected_cash_after
    assert info.buy_scale < 1.0 and projected_cash_after(CFG, ms, trade, Decimal(1)) >= 0
    assert all(reasons[a] == "cash_pro_rata" for a in ASSETS if unc[a])
    ratios = [trade[a] / unc[a] for a in ASSETS if unc[a]]
    assert max(ratios) - min(ratios) < 1e-3  # the same factor for every buy (up to lot rounding)


def test_explicit_no_trade_reasons():
    _, ms = state()
    at_target = {a: int((Decimal(repr(float(TGT[i]))) * ms.nav / ms.unit[a]).to_integral_value(rounding="ROUND_DOWN"))
                 for i, a in enumerate(ASSETS)}
    snap2 = LedgerSnapshot(Decimal("100000000.00"), at_target, CUT)
    ms2 = mark_state(CFG, MD, CALS, snap2, CUT)
    _, _, _, info = prorata_trade(CFG, ms2, snap2, _target_from(ms2, snap2), turnover_limit=0.5, multiplier=Decimal(1))
    assert info.no_trade_reason == "already_at_target"
    dark = pd.Timestamp("2024-03-14T20:15:00Z")  # NYSE_C untradable
    snap3, ms3 = state(cutoff=dark)
    only_c = np.array([0.0, 0.0, 0.3, 0.7])
    trade, _, _, info3 = prorata_trade(CFG, ms3, snap3, only_c, turnover_limit=0.5, multiplier=Decimal(1))
    assert trade["NYSE_C"] == 0 and info3.no_trade_reason == "no_feasible_nonzero_order"


def _target_from(ms, snap):
    risky = [float(Decimal(snap.holdings[a]) * ms.unit[a] / ms.nav) for a in ASSETS]
    return np.array([*risky, 1 - sum(risky)])


def test_an_untradable_asset_is_never_traded():
    snap, ms = state(cutoff=pd.Timestamp("2024-03-14T20:15:00Z"))
    trade, _, _, _ = prorata_trade(CFG, ms, snap, TGT, turnover_limit=0.5, multiplier=Decimal(1))
    assert trade["NYSE_C"] == 0 and trade["KRX_A"] > 0


def test_lots_larger_than_one_are_respected():
    assets = [a.model_copy(update={"lot_size": 50}) for a in CFG.universe.assets]
    cfg = CFG.model_copy(update={"universe": CFG.universe.model_copy(update={"assets": assets})})
    snap = LedgerSnapshot(Decimal("1000000000.00"), dict.fromkeys(ASSETS, 0), CUT)
    ms = mark_state(cfg, MD, CALS, snap, CUT)
    trade, _, _, _ = prorata_trade(cfg, ms, snap, TGT, turnover_limit=0.5, multiplier=Decimal(1))
    assert all(q % 50 == 0 for q in trade.values()) and sum(trade.values()) > 0


def test_policy_diagnostics_never_hide_the_turnover_scaling():
    from quant_pipeline.benchmarks import DecisionInput
    snap, ms = state()
    pol = make_policy(CFG, "equal_weight_periodic")
    plan, info = pol.decide(DecisionInput(1, snap, ms, 0, "b", "f", "D001", Decimal(1), CALS))
    assert plan.summary["prorata"]["scale"] == pytest.approx(0.5 / 0.9) and info["required_turnover"] == pytest.approx(0.9)
    assert plan.trial_hash == "benchmark:equal_weight_periodic" and plan.refit_id == "none"
