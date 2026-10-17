"""Bank of Korea ECOS adapter for the KRW overnight reference rate (spec 3.x ``krw_rfr_public``; OQ-050) and the live composite provider.

The spec asks for a public KRW overnight/reference risk-free-rate series, annual simple, ACT/365F, published one XKRX business day after its
date. FRED only carries monthly Korean short-term rates, which would break the five-business-day carry limit, so the daily overnight call rate
comes from ECOS (free API key, environment variable ``ECOS_API_KEY``). The statistic and item codes are configuration, not constants: they are
checked against ECOS's own item list (``RealEcosClient.item_names``) in the data workflow before a snapshot is accepted, and the series name is
recorded in the snapshot.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from .calendars import Calendars
from .collector import (
    RATE_COLUMNS,
    Provider,
    ProviderError,
    ProviderResponse,
    RateLimited,
    Request,
    SchemaDrift,
    SymbolNotFound,
    TransientError,
)

SOURCE_ID = "bok_ecos_call_overnight"
SOURCE_REVISION = "unversioned"
DEFAULT_STAT_CODE = "817Y002"  # ECOS daily market interest rates
DEFAULT_ITEM_CODE = "010101000"  # call rate, 1 day, all transactions (confirmed by name before use)
EXPECTED_ITEM_WORDS = ("콜", "call")  # the item name must mention the call (overnight) rate
BASE_URL = "https://ecos.bok.or.kr/api"
MAX_ROWS = 100000
NO_DATA_CODES = {"INFO-200"}
RATE_LIMIT_CODES = {"ERROR-602", "INFO-300"}


class EcosClient(Protocol):
    def series(self, stat_code: str, item_code: str, start: str, end: str) -> list[dict[str, str]]: ...


class RealEcosClient:
    """The only network-touching piece. The API key is part of the URL path, so no URL or request detail may reach a message or a log."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self._key = api_key or os.environ.get("ECOS_API_KEY", "")
        self.timeout = timeout
        if not self._key:
            raise ProviderError("ECOS_API_KEY is not set")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{BASE_URL}/{path}".replace("{KEY}", urllib.parse.quote(self._key, safe=""))
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimited("ECOS rate limit") from None
            raise TransientError(f"ECOS HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise TransientError(f"ECOS request failed: {type(e).__name__}") from None

    @staticmethod
    def _raise_for(payload: dict[str, Any]) -> None:
        result = payload.get("RESULT")
        if result is None:
            return
        code = str(result.get("CODE", ""))
        if code in NO_DATA_CODES:
            raise SymbolNotFound("ECOS returned no data for the series and range")
        if code in RATE_LIMIT_CODES:
            raise RateLimited(f"ECOS {code}")
        raise ProviderError(f"ECOS error {code}")  # the message text is not forwarded: it can echo request details

    def series(self, stat_code: str, item_code: str, start: str, end: str) -> list[dict[str, str]]:
        payload = self._get(f"StatisticSearch/{{KEY}}/json/kr/1/{MAX_ROWS}/{stat_code}/D/{start}/{end}/{item_code}")
        self._raise_for(payload)
        block = payload.get("StatisticSearch")
        if not isinstance(block, dict) or "row" not in block:
            raise SchemaDrift("ECOS response has no StatisticSearch.row")
        return list(block["row"])

    def item_names(self, stat_code: str) -> dict[str, str]:
        """item code -> item name, used to confirm that the configured item really is the overnight call rate."""
        payload = self._get(f"StatisticItemList/{{KEY}}/json/kr/1/1000/{stat_code}")
        self._raise_for(payload)
        block = payload.get("StatisticItemList")
        if not isinstance(block, dict) or "row" not in block:
            raise SchemaDrift("ECOS response has no StatisticItemList.row")
        return {r["ITEM_CODE"]: r["ITEM_NAME"] for r in block["row"]}


def rates_frame(rows: list[dict[str, str]], *, cals: Calendars, publication_delay_business_days: int, item_name: str | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Daily percent quotes to the canonical rate table; a rate dated D is available ``publication_delay_business_days`` XKRX sessions later."""
    if not rows:
        raise SymbolNotFound("ECOS returned no rows")
    for r in rows:
        if "TIME" not in r or "DATA_VALUE" not in r:
            raise SchemaDrift("ECOS row lacks TIME or DATA_VALUE")
    df = pd.DataFrame({"date": pd.to_datetime([r["TIME"] for r in rows], format="%Y%m%d"),
                       "pct": pd.to_numeric([r["DATA_VALUE"] for r in rows], errors="coerce")})
    if df["date"].duplicated().any():
        raise SchemaDrift("ECOS returned duplicate dates")
    df = df.sort_values("date")
    usable = df["pct"].notna()
    sessions = cals.sessions("XKRX", df["date"].iloc[0], df["date"].iloc[-1] + pd.Timedelta(days=20))
    in_session = df["date"].isin(sessions)
    dropped = {"unparseable_rows": int((~usable).sum()), "non_session_rows": int((usable & ~in_session).sum())}
    df = df[usable & in_session]
    if df.empty:
        raise SymbolNotFound("no usable ECOS rows")
    if (df["pct"].abs() > 100).any():
        raise SchemaDrift("ECOS rate outside any plausible percent range")
    exchange = cals.cals["XKRX"]
    avail = [exchange.session_offset(d, publication_delay_business_days) for d in df["date"]]
    fmt_day = lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%dT00:00:00Z")
    out = pd.DataFrame({
        "rate_date": df["date"].dt.strftime("%Y-%m-%d").to_numpy(), "event_timestamp_utc": [fmt_day(d) for d in df["date"]],
        "available_at_utc": [fmt_day(a) for a in avail], "annual_rate": (df["pct"] / 100.0).to_numpy(float), "quote_basis": "annual_simple",
        "day_count": "ACT/365F", "source_id": SOURCE_ID})[RATE_COLUMNS]
    return out, dropped


@dataclass(frozen=True)
class EcosSeries:
    stat_code: str = DEFAULT_STAT_CODE
    item_code: str = DEFAULT_ITEM_CODE


class EcosProvider(Provider):
    name = "ecos"
    adapter_version = "1"
    license = ("Bank of Korea ECOS Open API, public statistics used under the service's terms with an issued API key; "
               "attribution to the Bank of Korea required (user-attested)")

    def __init__(self, client: EcosClient, cals: Calendars, *, publication_delay_business_days: int, series: EcosSeries | None = None):
        self.client, self.cals, self.delay_days = client, cals, publication_delay_business_days
        self.series = series or EcosSeries()
        self.diagnostics: dict[str, Any] = {}

    def fetch_rates(self, req: Request) -> ProviderResponse:
        rows = self.client.series(self.series.stat_code, self.series.item_code, req.start.replace("-", ""), req.end.replace("-", ""))
        frame, dropped = rates_frame(rows, cals=self.cals, publication_delay_business_days=self.delay_days)
        self.diagnostics = {**dropped, "rows": len(frame), "first": frame["rate_date"].iloc[0], "last": frame["rate_date"].iloc[-1]}
        return ProviderResponse(frame, {"source": SOURCE_ID, **dropped}, SOURCE_REVISION)


class LiveProvider(Provider):
    """Prices, total return, actions, metadata and FX from one provider and rates from another, presented as one provider to the collector."""

    adapter_version = "1"

    def __init__(self, market: Provider, rates: Provider):
        self.market, self.rates = market, rates
        self.name = f"{market.name}+{rates.name}"
        self.license = f"{market.license}; {rates.license}"

    def fetch_metadata(self, req: Request) -> ProviderResponse:
        return self.market.fetch_metadata(req)

    def fetch_raw_prices(self, req: Request) -> ProviderResponse:
        return self.market.fetch_raw_prices(req)

    def fetch_total_return(self, req: Request) -> ProviderResponse:
        return self.market.fetch_total_return(req)

    def fetch_corporate_actions(self, req: Request) -> ProviderResponse:
        return self.market.fetch_corporate_actions(req)

    def fetch_fx(self, req: Request) -> ProviderResponse:
        return self.market.fetch_fx(req)

    def fetch_rates(self, req: Request) -> ProviderResponse:
        return self.rates.fetch_rates(req)
