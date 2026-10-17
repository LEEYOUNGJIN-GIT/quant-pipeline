"""Assembly of the live provider (Yahoo prices and FX, ECOS overnight rate) from a profile and its universe (OQ-050)."""
from __future__ import annotations

from .calendars import Calendars
from .collector import Provider
from .config import Asset, Config
from .ecos_adapter import EcosClient, EcosProvider, EcosSeries, LiveProvider, RealEcosClient
from .yfinance_adapter import AssetBinding, RealYahooClient, YahooClient, YahooProvider


def build_live_provider(cfg: Config, assets: list[Asset], *, yahoo: YahooClient | None = None, ecos: EcosClient | None = None,
                        series: EcosSeries | None = None) -> Provider:
    """The clients are injectable so the whole path can be exercised offline; the defaults are the real network clients."""
    cals = Calendars.load(["XKRX", "XNYS"])
    bindings = {a.asset_id: AssetBinding(a.exchange, a.currency) for a in assets}
    av = cfg.availability
    market = YahooProvider(yahoo or RealYahooClient(), bindings, cals, price_delay=av.price_publication_delay, fx_delay=av.fx_publication_delay)
    rates = EcosProvider(ecos or RealEcosClient(), cals, publication_delay_business_days=av.rate_publication_delay_business_days, series=series)
    return LiveProvider(market, rates)
