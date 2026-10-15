"""Snapshot/collector separation and fetch reporting (spec 4.6, check ``INV-FETCH-001``).

The collector is an optional preparation step. It asks a provider adapter for the six capabilities, caches every response by a
content-addressed key, validates the result and only then materializes an immutable snapshot directory (``manifest.json`` plus
``fetch_report.json``, written atomically). Normal ``run``/``evaluate``/``holdout`` never import this module's network paths.

Rules enforced here:

* Partial success is a failure: missing symbols, unsupported capabilities, rate-limit exhaustion, schema drift, ambiguous
  adjustments and every retry are listed per item, and no snapshot is written.
* A failed fetch never overwrites a valid cache entry or snapshot (failure reports go to a sibling ``<output>.failed`` directory).
* Credentials are never logged or hashed: request parameters carrying secret-looking keys are rejected, error text is redacted.
* Retry jitter is derived from the ``provider_jitter`` stream and the request key, so it does not depend on thread scheduling.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Asset, Collector
from .hashing import hash_json_obj

CAPABILITIES = ("metadata", "raw_prices", "total_return", "corporate_actions", "fx", "rates")
ADJUSTMENT_MODE = {"metadata": "none", "raw_prices": "none", "total_return": "gross_total_return_index", "corporate_actions": "none",
                   "fx": "none", "rates": "none"}
PRICE_COLUMNS = ["instrument_id", "source_timestamp", "event_timestamp_utc", "available_at_utc", "exchange_session", "field", "value", "currency",
                 "unit", "source_id", "source_revision", "is_final", "is_stale", "quality_flag"]
ACTION_COLUMNS = ["action_id", "instrument_id", "action_type", "declared_at_utc", "available_at_utc", "ex_session_date", "ex_timestamp_utc",
                  "pay_session_date", "pay_timestamp_utc", "ratio_num", "ratio_den", "cash_amount", "cash_currency"]
FX_COLUMNS = ["pair_id", "event_timestamp_utc", "available_at_utc", "value", "quote_convention", "source_id"]
RATE_COLUMNS = ["rate_date", "event_timestamp_utc", "available_at_utc", "annual_rate", "quote_basis", "day_count", "source_id"]
METADATA_COLUMNS = ["symbol", "currency"]
SCHEMAS: dict[str, tuple[list[str], list[str]]] = {  # file stem -> (columns, primary key)
    "prices_raw": (PRICE_COLUMNS, ["instrument_id", "event_timestamp_utc"]),
    "prices_total_return": (PRICE_COLUMNS, ["instrument_id", "event_timestamp_utc"]),
    "corporate_actions": (ACTION_COLUMNS, ["action_id"]),
    "fx_krw_per_usd": (FX_COLUMNS, ["pair_id", "event_timestamp_utc"]),
    "rates_krw_rfr": (RATE_COLUMNS, ["event_timestamp_utc"]),
}
SNAPSHOT_FILES = {"raw_prices": "prices_raw", "total_return": "prices_total_return", "corporate_actions": "corporate_actions", "fx": "fx_krw_per_usd", "rates": "rates_krw_rfr"}
SUPPORTED_ACTIONS = frozenset({"split", "cash_dividend"})
SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|secret|passw(or)?d|authorization|credential|cookie)")
SECRET_VALUE = re.compile(r"(?i)\b(api[_-]?key|token|secret|passw(?:or)?d|authorization|bearer)\b(\s*[=:]\s*|\s+)\S+")
PLACEHOLDERS = ("<snapshot_id>", "{snapshot_id}")
UNIVERSE_FILE, HOLDINGS_FILE = "universe.yaml", "initial_holdings.csv"


# ----------------------------------------------------------------------------------------------------------- errors
class UnsupportedCapability(Exception):
    """The provider explicitly does not offer this capability (reported per item, never silently skipped)."""


class ProviderError(Exception):
    kind = "provider_error"


class SymbolNotFound(ProviderError):
    kind = "missing_symbol"


class SchemaDrift(ProviderError):
    kind = "schema_drift"


class RateLimited(ProviderError):
    kind = "rate_limit_exhausted"
    retryable = True


class TransientError(ProviderError):
    kind = "retries_exhausted"
    retryable = True


class SnapshotError(Exception):
    """A snapshot directory is missing files, has a hash mismatch, or conflicts with an existing immutable snapshot."""


def redact(text: str) -> str:
    """Remove credential-looking fragments from provider text before it reaches a report."""
    return SECRET_VALUE.sub(lambda m: f"{m.group(1)}=<redacted>", text)[:500]


# ----------------------------------------------------------------------------------------------------------- provider contract
@dataclass(frozen=True)
class Request:
    capability: str
    symbol: str
    start: str
    end: str
    interval: str = "1d"
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def adjustment_mode(self) -> str:
        return ADJUSTMENT_MODE[self.capability]


@dataclass
class ProviderResponse:
    payload: pd.DataFrame
    status: dict[str, Any] = field(default_factory=dict)  # e.g. {"http_status": 200}
    revision: str | None = None


class Provider:
    """Adapter contract: the six ``fetch_*`` methods return a canonical-schema frame or raise ``UnsupportedCapability``."""

    name = "provider"
    adapter_version = "0"
    license: str | None = None

    def fetch_metadata(self, req: Request) -> ProviderResponse:
        raise UnsupportedCapability("fetch_metadata")

    def fetch_raw_prices(self, req: Request) -> ProviderResponse:
        raise UnsupportedCapability("fetch_raw_prices")

    def fetch_total_return(self, req: Request) -> ProviderResponse:
        raise UnsupportedCapability("fetch_total_return")

    def fetch_corporate_actions(self, req: Request) -> ProviderResponse:
        raise UnsupportedCapability("fetch_corporate_actions")

    def fetch_fx(self, req: Request) -> ProviderResponse:
        raise UnsupportedCapability("fetch_fx")

    def fetch_rates(self, req: Request) -> ProviderResponse:
        raise UnsupportedCapability("fetch_rates")

    def fetch(self, req: Request) -> ProviderResponse:
        return getattr(self, f"fetch_{req.capability}")(req)


class YFinanceProvider(Provider):
    """Registered so the CLI contract matches the spec, but every capability is explicitly unsupported in this build.

    Mapping yfinance output to the point-in-time schema (availability, official sessions, ex/pay dates, adjustment convention) was not
    validated against the live service, so no unverified mapping is shipped (open-questions OQ-036).
    """

    name = "yfinance"
    adapter_version = "0-unsupported"
    license = None


class LocalFilesProvider(Provider):
    """Offline provider over a directory of canonical parquet files (used for network-off verification and for re-materializing snapshots)."""

    name = "local_files"
    adapter_version = "1"

    def __init__(self, directory: Path, license: str):
        self.directory, self.license = Path(directory), license

    def _frame(self, stem: str) -> pd.DataFrame:
        path = self.directory / f"{stem}.parquet"
        if not path.exists():
            raise UnsupportedCapability(f"{stem}.parquet is not present in {self.directory.name}")
        return pd.read_parquet(path)

    @staticmethod
    def _window(df: pd.DataFrame, col: str, req: Request) -> pd.DataFrame:
        day = pd.to_datetime(df[col], utc=True).dt.strftime("%Y-%m-%d")
        return df[(day >= req.start) & (day <= req.end)]

    def fetch_metadata(self, req: Request) -> ProviderResponse:
        prices = self._frame("prices_raw")
        rows = prices[prices.instrument_id == req.symbol]
        if rows.empty:
            raise SymbolNotFound(req.symbol)
        return ProviderResponse(pd.DataFrame({"symbol": [req.symbol], "currency": [rows.currency.iloc[0]]}), {"source": "local"})

    def _by_symbol(self, stem: str, symbol_col: str, time_col: str, req: Request) -> ProviderResponse:
        df = self._frame(stem)
        rows = df[df[symbol_col] == req.symbol]
        if rows.empty and stem.startswith("prices"):
            raise SymbolNotFound(req.symbol)
        return ProviderResponse(self._window(rows, time_col, req).reset_index(drop=True), {"source": "local"})

    def fetch_raw_prices(self, req: Request) -> ProviderResponse:
        return self._by_symbol("prices_raw", "instrument_id", "event_timestamp_utc", req)

    def fetch_total_return(self, req: Request) -> ProviderResponse:
        return self._by_symbol("prices_total_return", "instrument_id", "event_timestamp_utc", req)

    def fetch_corporate_actions(self, req: Request) -> ProviderResponse:
        return self._by_symbol("corporate_actions", "instrument_id", "ex_timestamp_utc", req)

    def fetch_fx(self, req: Request) -> ProviderResponse:
        return self._by_symbol("fx_krw_per_usd", "pair_id", "event_timestamp_utc", req)

    def fetch_rates(self, req: Request) -> ProviderResponse:
        return ProviderResponse(self._window(self._frame("rates_krw_rfr"), "event_timestamp_utc", req).reset_index(drop=True), {"source": "local"})


def assets_from_manifest(universe_manifest: Path) -> list[Asset]:
    """Explicit asset records of a research universe manifest (validation of size/exchanges stays with ``data validate``)."""
    from .config import load_yaml_strict

    records = load_yaml_strict(Path(universe_manifest).read_text(encoding="utf-8"))["assets"]
    return [Asset.model_validate(r) for r in records]


def make_provider(name: str, *, source_dir: Path | None = None, license: str | None = None) -> Provider:
    if name == "yfinance":
        return YFinanceProvider()
    if name == "local_files":
        if source_dir is None or not license:
            raise ValueError("provider local_files needs --source-dir and --license")
        return LocalFilesProvider(source_dir, license)
    raise ValueError(f"unknown provider {name!r}; available: yfinance, local_files")


# ----------------------------------------------------------------------------------------------------------- atomic files
def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_parquet_stable(df: pd.DataFrame, path: Path) -> None:
    """Deterministic parquet bytes for a given frame (no pandas metadata, fixed compression); the writer version still belongs to the lock."""
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False).replace_schema_metadata(None)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    pq.write_table(table, tmp, compression="snappy", version="2.6", use_dictionary=True)
    os.replace(tmp, path)


def logical_hash(path: Path, keys: list[str]) -> str:
    table = pq.read_table(path).sort_by([(k, "ascending") for k in keys]).replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    return "sha256:" + hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def raw_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------- cache
def cache_key(provider: Provider, req: Request) -> str:
    """``sha256(provider, adapter_version, request_parameters, symbol, interval, start, end, adjustment_mode)`` (spec 4.6)."""
    bad = [k for k in req.params if SECRET_KEY.search(k)]
    if bad:
        raise ValueError(f"request parameters must not carry credentials: {bad}")
    return hash_json_obj({"provider": provider.name, "adapter_version": provider.adapter_version,
                          "request_parameters": {"capability": req.capability, **req.params}, "symbol": req.symbol, "interval": req.interval,
                          "start": req.start, "end": req.end, "adjustment_mode": req.adjustment_mode}).removeprefix("sha256:")


class ProviderCache:
    """Content-addressed response cache: ``<key>.parquet`` plus ``<key>.meta.json`` (meta written last, so meta implies data)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.parquet", self.root / f"{key}.meta.json"

    def get(self, key: str) -> tuple[ProviderResponse | None, str]:
        """Return ``(response, state)`` with state ``hit``, ``miss`` or ``corrupt`` (a corrupt entry is ignored, never trusted)."""
        data, meta = self._paths(key)
        if not (data.exists() and meta.exists()):
            return None, "miss"
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if raw_hash(data) != info["content_hash"]:
                return None, "corrupt"
            return ProviderResponse(pd.read_parquet(data), info["status"], info.get("provider_revision")), "hit"
        except (ValueError, KeyError, OSError):
            return None, "corrupt"

    def put(self, key: str, resp: ProviderResponse, fetched_at: str) -> None:
        data, meta = self._paths(key)
        self.root.mkdir(parents=True, exist_ok=True)
        write_parquet_stable(resp.payload, data)
        info = {"fetched_at_utc": fetched_at, "status": resp.status, "provider_revision": resp.revision, "content_hash": raw_hash(data),
                "rows": len(resp.payload)}
        write_atomic(meta, (json.dumps(info, sort_keys=True, indent=1) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------------------------------------------- rate limit and retries
class RateLimiter:
    """At most ``requests_per_minute`` request starts, thread-safe, with an injectable clock and sleep."""

    def __init__(self, requests_per_minute: int, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.interval, self.clock, self.sleep = 60.0 / requests_per_minute, clock, sleep
        self._next, self._lock = None, threading.Lock()

    def acquire(self) -> float:
        with self._lock:
            now = self.clock()
            start = now if self._next is None else max(now, self._next)
            self._next = start + self.interval
        wait = start - now
        if wait > 0:
            self.sleep(wait)
        return wait


def jitter(seed: int, key: str, attempt: int) -> float:
    """Deterministic jitter in [0, 1) from the ``provider_jitter`` stream, the request key and the attempt number."""
    return random.Random(f"{seed}:{key}:{attempt}").random()


@dataclass
class Item:
    capability: str
    symbol: str
    status: str = "pending"  # success | failed
    cache: str = "miss"
    error_kind: str | None = None
    error: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    rows: int = 0


# ----------------------------------------------------------------------------------------------------------- collection
@dataclass
class FetchReport:
    status: str
    snapshot_id: str | None
    provider: dict[str, Any]
    request: dict[str, Any]
    items: list[Item]
    issues: list[dict[str, str]]
    cache: dict[str, int]
    fetched_at_utc: str

    def failures(self) -> list[dict[str, Any]]:
        out = [{"capability": i.capability, "symbol": i.symbol, "kind": i.error_kind, "error": i.error, "attempts": len(i.attempts)}
               for i in self.items if i.status == "failed"]
        return out + [{"capability": x.get("capability", ""), "symbol": x.get("symbol", ""), "kind": x["kind"], "error": x["detail"], "attempts": 0}
                      for x in self.issues]

    def to_json(self) -> bytes:
        body = {"schema": "fetch-report-1", "status": self.status, "snapshot_id": self.snapshot_id, "provider": self.provider,
                "request": self.request, "cache": self.cache, "fetched_at_utc": self.fetched_at_utc, "failures": self.failures(),
                "items": [{"capability": i.capability, "symbol": i.symbol, "status": i.status, "cache": i.cache, "rows": i.rows,
                           "error_kind": i.error_kind, "error": i.error, "attempts": i.attempts} for i in self.items]}
        return (json.dumps(body, sort_keys=True, indent=1) + "\n").encode("utf-8")


def _requests(assets: list[Asset], fx_pair: str, rates_symbol: str, start: str, end: str) -> list[Request]:
    reqs = []
    for a in assets:
        for cap in ("metadata", "raw_prices", "total_return", "corporate_actions"):
            reqs.append(Request(cap, a.asset_id, start, end))
    reqs.append(Request("fx", fx_pair, start, end))
    reqs.append(Request("rates", rates_symbol, start, end))
    return reqs


def _fetch_one(req: Request, provider: Provider, cache: ProviderCache, cfg: Collector, limiter: RateLimiter, sleep: Callable[[float], None],
               seed: int, now: Callable[[], str]) -> tuple[Item, ProviderResponse | None]:
    item = Item(req.capability, req.symbol)
    key = cache_key(provider, req)
    cached, state = cache.get(key)
    item.cache = state
    if cached is not None:
        item.status, item.rows = "success", len(cached.payload)
        return item, cached
    for attempt in range(cfg.retries + 1):
        limiter.acquire()
        try:
            resp = provider.fetch(req)
        except UnsupportedCapability as e:
            item.status, item.error_kind, item.error = "failed", "unsupported_capability", redact(str(e))
            return item, None
        except ProviderError as e:
            item.attempts.append({"attempt": attempt + 1, "error": e.kind, "detail": redact(str(e))})
            if getattr(e, "retryable", False) and attempt < cfg.retries:
                delay = cfg.backoff_seconds * (2 ** attempt) + jitter(seed, key, attempt) * cfg.backoff_seconds
                item.attempts[-1]["backoff_seconds"] = round(delay, 6)
                sleep(delay)
                continue
            item.status, item.error_kind, item.error = "failed", e.kind, redact(str(e))
            return item, None
        except Exception as e:  # noqa: BLE001 - an unexpected provider failure is reported per item, never swallowed or cached
            item.attempts.append({"attempt": attempt + 1, "error": "provider_error", "detail": redact(f"{type(e).__name__}: {e}")})
            item.status, item.error_kind, item.error = "failed", "provider_error", redact(f"{type(e).__name__}: {e}")
            return item, None
        item.attempts.append({"attempt": attempt + 1, "error": None})
        if not isinstance(resp.payload, pd.DataFrame):
            item.status, item.error_kind, item.error = "failed", "schema_drift", "provider returned a non-frame payload"
            return item, None
        cache.put(key, resp, now())  # only a successful response ever reaches the cache
        item.status, item.rows = "success", len(resp.payload)
        return item, resp
    raise AssertionError("unreachable")


def _columns_for(capability: str) -> list[str] | None:
    return {"raw_prices": PRICE_COLUMNS, "total_return": PRICE_COLUMNS, "corporate_actions": ACTION_COLUMNS, "fx": FX_COLUMNS, "rates": RATE_COLUMNS,
            "metadata": METADATA_COLUMNS}.get(capability)


def _validate(items_resp: list[tuple[Item, ProviderResponse | None]], assets: list[Asset], provider: Provider, start: str, end: str) -> list[dict[str, str]]:
    """Collector output is not trusted until schema, availability, license, raw/total-return, corporate-action and range checks pass."""
    issues: list[dict[str, str]] = []

    def issue(kind: str, item: Item, detail: str) -> None:
        issues.append({"kind": kind, "capability": item.capability, "symbol": item.symbol, "detail": detail})

    if not provider.license or provider.license.strip().lower() in {"unknown", "none", "unlicensed"}:
        issues.append({"kind": "license_unknown", "capability": "", "symbol": "", "detail": f"provider {provider.name} declares no usable license"})
    ccy = {a.asset_id: a.currency for a in assets}
    for item, resp in items_resp:
        if resp is None:
            continue
        df, cols = resp.payload, _columns_for(item.capability)
        if cols is not None and list(df.columns) != cols:
            issue("schema_drift", item, f"columns {list(df.columns)} differ from the contract {cols}")
            continue
        if item.capability in ("raw_prices", "total_return"):
            want = "close" if item.capability == "raw_prices" else "total_return_index"
            fields = set(df["field"])
            if "adjusted_close" in fields or (fields and fields != {want}):
                issue("ambiguous_adjustment", item, f"field values {sorted(fields)}; expected only {want!r} (a single adjusted_close is ambiguous)")
            if set(df["instrument_id"]) - {item.symbol}:
                issue("schema_drift", item, f"rows for other instruments {sorted(set(df['instrument_id']) - {item.symbol})}")
        if item.capability in ("raw_prices", "total_return", "fx", "rates"):
            ev = pd.to_datetime(df["event_timestamp_utc"], utc=True, errors="coerce")
            av = pd.to_datetime(df["available_at_utc"], utc=True, errors="coerce")
            if ev.isna().any() or av.isna().any():
                issue("schema_drift", item, "unparseable timestamps")
            elif (av < ev).any():
                issue("availability", item, f"{int((av < ev).sum())} rows are available before their event time")
            else:
                days = ev.dt.strftime("%Y-%m-%d")
                if ((days < start) | (days > end)).any():
                    issue("outside_requested_range", item, f"rows outside {start}..{end}")
            if df.empty:
                issue("no_data", item, "the provider returned no rows for the requested range")
        if item.capability in ("raw_prices", "total_return") and not df.empty and bool((pd.to_numeric(df["value"], errors="coerce") <= 0).any()):
            issue("schema_drift", item, "non-positive or non-numeric values")
        if item.capability == "corporate_actions" and set(df.get("action_type", [])) - SUPPORTED_ACTIONS:
            issue("unsupported_action", item, f"types {sorted(set(df['action_type']) - SUPPORTED_ACTIONS)}")
        if item.capability == "metadata" and not df.empty and item.symbol in ccy and df["currency"].iloc[0] != ccy[item.symbol]:
            issue("metadata_mismatch", item, f"provider currency {df['currency'].iloc[0]} differs from the universe record {ccy[item.symbol]}")
    return issues


def collect(assets: list[Asset], *, fx_pair: str, rates_symbol: str, collector: Collector, provider: Provider, start: str, end: str,
            output: Path, cache_root: Path, jitter_seed: int, clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep, now: Callable[[], str] | None = None,
            universe_source: Path | None = None, holdings_source: Path | None = None) -> FetchReport:
    """Fetch every capability for every asset, validate, and materialize ``output`` only on complete success.

    With ``universe_source`` the snapshot also carries ``universe.yaml`` and ``initial_holdings.csv`` (``holdings_source`` or the default
    100 % KRW cash record), so that every file a research profile reads is covered by the manifest hash.
    """
    now = now or (lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    if not collector.fail_on_partial:
        raise ValueError("spec 4.6: partial success is a failure; collector.fail_on_partial must be true")
    cache, limiter = ProviderCache(cache_root), RateLimiter(collector.requests_per_minute, clock, sleep)
    reqs = _requests(assets, fx_pair, rates_symbol, start, end)
    with ThreadPoolExecutor(max_workers=max(1, collector.max_concurrency)) as pool:  # results keep request order, so the report is deterministic
        results = list(pool.map(lambda r: _fetch_one(r, provider, cache, collector, limiter, sleep, jitter_seed, now), reqs))
    issues = _validate(results, assets, provider, start, end)
    items = [i for i, _ in results]
    hits = sum(1 for i in items if i.cache == "hit")
    report = FetchReport("failed", None, {"name": provider.name, "adapter_version": provider.adapter_version, "license": provider.license},
                         {"start": start, "end": end, "assets": [a.asset_id for a in assets], "fx_pair": fx_pair, "interval": "1d"},
                         items, issues, {"hits": hits, "misses": len(items) - hits}, now())
    if any(i.status == "failed" for i in items) or issues:
        _write_failure(report, output)
        return report
    frames = {stem: [] for stem in SCHEMAS}
    for item, resp in results:
        if resp is not None and item.capability in SNAPSHOT_FILES:
            frames[SNAPSHOT_FILES[item.capability]].append(resp.payload)
    tables = {stem: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SCHEMAS[stem][0])) for stem, parts in frames.items()}
    extra: dict[str, bytes] = {}
    if universe_source is not None:
        extra[UNIVERSE_FILE] = universe_source.read_bytes()
        extra[HOLDINGS_FILE] = holdings_source.read_bytes() if holdings_source is not None else default_initial_holdings(assets, start).encode("utf-8")
    return _materialize(report, tables, provider, output, extra)


def default_initial_holdings(assets: list[Asset], start: str, cash_krw: str = "1000000000.00") -> str:
    """Spec 13/15 default: 100 % KRW cash, no risky positions, marked at the start of the collected range (OQ-038)."""
    as_of = f"{start}T00:00:00Z"
    rows = [f"CASH_KRW,0,KRW,{cash_krw},{as_of}"] + [f"{a.asset_id},0,{a.currency},,{as_of}" for a in assets]
    return "asset_id,quantity,currency,cash_amount,as_of_utc\n" + "\n".join(rows) + "\n"


def _materialize(report: FetchReport, tables: dict[str, pd.DataFrame], provider: Provider, output: Path,
                 extra: dict[str, bytes] | None = None) -> FetchReport:
    tmp = output.parent / f".{output.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    for stem, df in tables.items():
        cols, keys = SCHEMAS[stem]
        df = df[cols].sort_values(keys, kind="stable")
        path = tmp / f"{stem}.parquet"
        write_parquet_stable(df, path)
        files[path.name] = {"raw_sha256": raw_hash(path), "logical_sha256": logical_hash(path, keys), "rows": len(df)}
    for name, data in (extra or {}).items():  # text inputs: raw and logical hashes coincide (as in the fixture manifest)
        write_atomic(tmp / name, data)
        files[name] = {"raw_sha256": raw_hash(tmp / name), "logical_sha256": raw_hash(tmp / name), "rows": data.count(b"\n")}
    manifest = {"manifest_version": 1, "snapshot_kind": "research", "provider": report.provider, "request": report.request, "files": files}
    snapshot_id = hash_json_obj(manifest)
    report.status, report.snapshot_id = "success", snapshot_id
    write_atomic(tmp / "manifest.json", (json.dumps({**manifest, "snapshot_id": snapshot_id}, sort_keys=True, indent=1) + "\n").encode("utf-8"))
    write_atomic(tmp / "fetch_report.json", report.to_json())
    final = Path(str(output).replace(PLACEHOLDERS[0], snapshot_id.removeprefix("sha256:")).replace(PLACEHOLDERS[1], snapshot_id.removeprefix("sha256:")))
    try:
        if final.exists():
            existing = verify_snapshot(final)
            shutil.rmtree(tmp)
            if existing != snapshot_id:
                raise SnapshotError(f"{final} already holds snapshot {existing}; snapshots are immutable and {snapshot_id} was not written")
            return report  # identical content: nothing to change
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, final)
    except SnapshotError:
        report.status, report.snapshot_id = "failed", None
        report.issues.append({"kind": "snapshot_conflict", "capability": "", "symbol": "", "detail": f"{final} holds a different snapshot or is unreadable"})
        _write_failure(report, final)
        shutil.rmtree(tmp, ignore_errors=True)
    return report


def _write_failure(report: FetchReport, output: Path) -> None:
    out = Path(str(output).replace(PLACEHOLDERS[0], "unknown").replace(PLACEHOLDERS[1], "unknown"))
    write_atomic(out.parent / f"{out.name}.failed" / "fetch_report.json", report.to_json())  # a sibling: an existing snapshot is never touched


def verify_snapshot(directory: Path) -> str:
    """Re-hash every referenced file and the manifest; return the snapshot id or raise ``SnapshotError``."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SnapshotError(f"{directory}: unreadable manifest.json ({type(e).__name__})") from e
    claimed = manifest.pop("snapshot_id", None)
    if hash_json_obj(manifest) != claimed:
        raise SnapshotError(f"{directory}: snapshot_id does not equal the manifest's canonical hash")
    for name, meta in manifest["files"].items():
        path = directory / name
        if not path.exists():
            raise SnapshotError(f"{directory}: referenced file {name} is missing")
        if raw_hash(path) != meta["raw_sha256"]:
            raise SnapshotError(f"{directory}: raw hash mismatch for {name}")
        if path.suffix != ".parquet":
            continue  # text inputs: the raw hash above is also their logical hash
        keys = SCHEMAS[path.stem][1]
        if logical_hash(path, keys) != meta["logical_sha256"]:
            raise SnapshotError(f"{directory}: logical hash mismatch for {name}")
    return claimed
