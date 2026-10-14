"""Exact decimal quantity/cash ledger kernel (spec 10.1). One fill at a time, costs deducted exactly once.

Money is quantized to ``money_decimal_places`` per posting; quantities are integers (lot-sized upstream).
Rounding mode is not specified by the spec text reviewed so far: ROUND_HALF_EVEN is a provisional choice
recorded in open-questions.md (OQ-001).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal

from .fx import krw_value

BPS = Decimal(10_000)


def quantize_money(x: Decimal, places: int = 2) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)


def trade_cost(gross: Decimal, rates: CostRates, is_usd: bool, is_sell: bool, places: int = 2,
               sell_tax_bps: Decimal | None = None) -> Decimal:
    """Sum of explicitly quantized cost postings (commission, spread/slippage, sell tax, FX conversion; spec 10.1).

    Shared by the ledger (realized cost) and the rebalance plan (projected cost) so they cannot diverge.
    """
    tax_bps = rates.sell_tax_bps if sell_tax_bps is None else sell_tax_bps
    commission = max(quantize_money(gross * rates.commission_bps / BPS, places), rates.minimum_fee_base)
    spread = quantize_money(gross * rates.half_spread_slippage_bps / BPS, places)
    tax = quantize_money(gross * tax_bps / BPS, places) if is_sell else Decimal(0)
    fxc = quantize_money(gross * rates.fx_conversion_bps / BPS, places) if is_usd else Decimal(0)
    return commission + spread + tax + fxc


@dataclass(frozen=True)
class CostRates:
    commission_bps: Decimal
    half_spread_slippage_bps: Decimal
    sell_tax_bps: Decimal
    fx_conversion_bps: Decimal
    minimum_fee_base: Decimal = Decimal(0)

    def scaled(self, multiplier: Decimal) -> CostRates:
        """Cost multiplier scales realized rates (spec 10.2); the minimum fee is left unscaled."""
        return CostRates(
            self.commission_bps * multiplier,
            self.half_spread_slippage_bps * multiplier,
            self.sell_tax_bps * multiplier,
            self.fx_conversion_bps * multiplier,
            self.minimum_fee_base,
        )


@dataclass(frozen=True)
class Fill:
    asset_id: str
    currency: str
    quantity: int  # buy > 0, sell < 0
    raw_price: Decimal
    fx: Decimal | None  # KRW_per_USD at the fill timestamp; None for KRW assets
    sell_tax_bps: Decimal = Decimal(0)


@dataclass
class Ledger:
    cash: Decimal
    holdings: dict[str, int] = field(default_factory=dict)
    money_places: int = 2
    cumulative_costs: Decimal = Decimal(0)
    allow_negative_cash: bool = False

    def post_fill(self, fill: Fill, rates: CostRates) -> dict[str, Decimal]:
        if fill.quantity == 0:
            raise ValueError("zero-quantity fill")
        qty = Decimal(fill.quantity)
        gross = quantize_money(krw_value(abs(qty), fill.raw_price, fill.currency, fill.fx), self.money_places)
        cost = trade_cost(gross, rates, fill.currency == "USD", fill.quantity < 0, self.money_places, fill.sell_tax_bps)
        signed_notional = gross if fill.quantity > 0 else -gross
        new_cash = self.cash - signed_notional - cost
        if new_cash < 0 and not self.allow_negative_cash:
            raise ValueError(f"fill of {fill.asset_id} would make cash negative ({new_cash})")
        new_qty = self.holdings.get(fill.asset_id, 0) + fill.quantity
        if new_qty < 0:
            raise ValueError(f"fill of {fill.asset_id} would create a short position")
        # post atomically only after every check has passed
        self.cash = new_cash
        self.holdings[fill.asset_id] = new_qty
        self.cumulative_costs += cost
        return {"notional": gross, "cost": cost, "cash_after": new_cash}

    def nav(self, marks_krw: dict[str, Decimal]) -> Decimal:
        """NAV = cash + sum(quantity * KRW mark). ``marks_krw`` is the KRW value of one unit."""
        total = self.cash
        for asset, q in self.holdings.items():
            if q:
                total += Decimal(q) * marks_krw[asset]
        return quantize_money(total, self.money_places)
