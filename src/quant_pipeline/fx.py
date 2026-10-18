"""FX conversion and return decomposition (spec 4.4, 6.6). Quote: KRW_per_USD = KRW to buy USD 1."""
from __future__ import annotations

from decimal import Decimal


def krw_return(r_local: float, r_fx: float) -> float:
    """1 + r_KRW = (1 + r_local)(1 + r_fx); the interaction term r_local*r_fx is retained."""
    return r_local + r_fx + r_local * r_fx


def decompose_krw_return(r_local: float, r_fx: float) -> dict[str, float]:
    interaction = r_local * r_fx
    return {"local": r_local, "fx": r_fx, "interaction": interaction, "total": r_local + r_fx + interaction}


def krw_value(quantity: Decimal, raw_price: Decimal, currency: str, krw_per_usd: Decimal | None) -> Decimal:
    """KRW value of a position. USD assets require an explicit KRW_per_USD quote; KRW assets reject one."""
    if currency == "KRW":
        if krw_per_usd is not None:
            raise ValueError("KRW asset must not be converted with an FX quote")
        return quantity * raw_price
    if currency == "USD":
        if krw_per_usd is None or krw_per_usd <= 0:
            raise ValueError("USD asset needs a positive KRW_per_USD quote")
        return quantity * raw_price * krw_per_usd
    raise ValueError(f"unsupported currency {currency!r}")


def invert_quote(usd_per_krw: Decimal) -> Decimal:
    """Explicit reciprocal adapter; inverse inputs are otherwise disallowed (``allow_inverse_input: false``)."""
    if usd_per_krw <= 0:
        raise ValueError("quote must be positive")
    return Decimal(1) / usd_per_krw
