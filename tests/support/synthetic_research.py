"""Deterministic SYNTHETIC research-shaped snapshot for engineering tests (never research evidence).

Eight invented instruments (R00-R03 on XKRX, R100-R103 on XNYS), geometric Brownian motion without corporate actions, an FX path and a
flat rate series, written as the canonical tables a provider would return and materialized through the collector. Ids are disjoint from
the fixture's. The generator uses its own seed and never imports the production row map or optimizer.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_pipeline import collector as C
from quant_pipeline.calendars import Calendars, fmt
from quant_pipeline.config import Asset, Collector, load_config

ROOT = Path(__file__).resolve().parents[2]
KRX_IDS = [f"R{i:02d}" for i in range(4)]
NYSE_IDS = [f"R{100 + i}" for i in range(4)]
SEED = 20261015
SETTINGS = Collector(enabled=True, provider="local_files", cache_root="unused", requests_per_minute=6000, max_concurrency=4, retries=0,
                     backoff_seconds=0.0, fail_on_partial=True)


def asset_records() -> list[dict]:
    recs = []
    for ids, ex, ccy, tz in ((KRX_IDS, "XKRX", "KRW", "Asia/Seoul"), (NYSE_IDS, "XNYS", "USD", "America/New_York")):
        for a in ids:
            recs.append({"asset_id": a, "exchange": ex, "currency": ccy, "timezone": tz, "calendar": ex, "raw_price_field": "close",
                         "model_price_field": "total_return_index", "lot_size": 1, "quantity_precision": 0, "allow_short": False,
                         "benchmark_eligible": True, "max_weight": 0.30})
    return recs


def _price_rows(cals: Calendars, ex: str, ccy: str, unit: str, ids: list[str], start: str, end: str, rng: np.random.Generator, field: str,
                paths: dict[str, np.ndarray] | None) -> tuple[list[dict], dict[str, np.ndarray]]:
    sess = cals.sessions(ex, start, end)
    rows, out = [], {}
    for a in ids:
        if paths is None:
            r = rng.normal(0.0003, 0.012, size=len(sess))
            level = 100.0 * np.exp(np.cumsum(r))
        else:
            level = paths[a]
        out[a] = level
        for s, v in zip(sess, level, strict=True):
            close = cals.close(ex, s)
            rows.append({"instrument_id": a, "source_timestamp": fmt(close), "event_timestamp_utc": fmt(close), "available_at_utc": fmt(close + pd.Timedelta(minutes=15)),
                         "exchange_session": s.strftime("%Y-%m-%d"), "field": field,
                         "value": float(round(v * (700.0 if ex == "XKRX" else 1.0), 4 if ex == "XNYS" else 0)) if field == "close" else float(v),
                         "currency": ccy, "unit": unit, "source_id": "synthetic-research", "source_revision": "1", "is_final": True,
                         "is_stale": False, "quality_flag": "ok"})
    return rows, out


def source_tables(start: str, end: str) -> dict[str, pd.DataFrame]:
    cals = Calendars.load(["XKRX", "XNYS"])
    rng = np.random.default_rng(SEED)
    raw_rows, tr_rows = [], []
    for ids, ex, ccy, unit in ((KRX_IDS, "XKRX", "KRW", "KRW/share"), (NYSE_IDS, "XNYS", "USD", "USD/share")):
        rows, paths = _price_rows(cals, ex, ccy, unit, ids, start, end, rng, "close", None)
        raw_rows += rows
        trs, _ = _price_rows(cals, ex, ccy, "index", ids, start, end, rng, "total_return_index", paths)
        tr_rows += trs
    # FX trades every weekday (a quote at each XKRX and XNYS close), so an exchange holiday never leaves a quote older than the 72h limit
    fx = []
    level = 1300.0
    stamps = sorted({cals.close(ex, s) for ex in ("XKRX", "XNYS") for s in cals.sessions(ex, start, end)})
    for close in stamps:
        level *= float(np.exp(rng.normal(0.0, 0.003)))
        fx.append({"pair_id": "USDKRW", "event_timestamp_utc": fmt(close), "available_at_utc": fmt(close + pd.Timedelta(minutes=15)),
                   "value": round(level, 2), "quote_convention": "KRW_per_USD", "source_id": "synthetic-research"})
    for d in pd.bdate_range(start, end):  # a midday quote on weekdays when both exchanges are closed bridges multi-day holidays
        noon = pd.Timestamp(f"{d:%Y-%m-%d}T12:00:00Z")
        if not any(abs((noon - t).total_seconds()) < 36 * 3600 for t in stamps if t.date() == d.date()):
            level *= float(np.exp(rng.normal(0.0, 0.003)))
            fx.append({"pair_id": "USDKRW", "event_timestamp_utc": fmt(noon), "available_at_utc": fmt(noon + pd.Timedelta(minutes=15)),
                       "value": round(level, 2), "quote_convention": "KRW_per_USD", "source_id": "synthetic-research"})
    fx.sort(key=lambda r: r["event_timestamp_utc"])
    days = pd.bdate_range(start, end)
    rates = [{"rate_date": d.strftime("%Y-%m-%d"), "event_timestamp_utc": f"{d:%Y-%m-%d}T00:00:00Z", "available_at_utc": f"{d:%Y-%m-%d}T00:00:00Z",
              "annual_rate": 0.03, "quote_basis": "annual_simple", "day_count": "ACT/365F", "source_id": "krw_rfr_public"} for d in days]
    return {"prices_raw": pd.DataFrame(raw_rows), "prices_total_return": pd.DataFrame(tr_rows), "corporate_actions": pd.DataFrame(columns=C.ACTION_COLUMNS),
            "fx_krw_per_usd": pd.DataFrame(fx), "rates_krw_rfr": pd.DataFrame(rates)}


def build_research_repo(root: Path, start: str = "2016-01-04", end: str = "2023-12-29",
                        mutate: Callable[[dict[str, pd.DataFrame]], None] | None = None) -> tuple[Path, str]:
    """A repository root with the synthetic snapshot at data/snapshots/research/<id> and ``current`` pointing at it."""
    src = root / "source"
    src.mkdir(parents=True)
    tables = source_tables(start, end)
    if mutate is not None:
        mutate(tables)
    for stem, df in tables.items():
        df.to_parquet(src / f"{stem}.parquet", index=False)
    universe = root / "universe_source.yaml"
    universe.write_text(yaml.safe_dump({"assets": asset_records()}, sort_keys=False), encoding="utf-8")
    assets = [Asset.model_validate(r) for r in asset_records()]
    cfg = load_config(ROOT / "configs" / "research_8_12asset.yaml")
    out = root / "data" / "snapshots" / "research" / "<snapshot_id>"
    report = C.collect(assets, fx_pair=cfg.fx.pair_id, rates_symbol="krw_rfr", collector=SETTINGS,
                       provider=C.LocalFilesProvider(src, "synthetic test data; no third-party rights"), start=start, end=end, output=out,
                       cache_root=root / "cache", jitter_seed=1731, now=lambda: "2026-01-01T00:00:00Z", universe_source=universe, synthetic=True)
    if report.status != "success":
        raise RuntimeError(f"synthetic snapshot failed: {report.issues}")
    sid = report.snapshot_id.removeprefix("sha256:")
    (root / "data" / "snapshots" / "research" / "current").symlink_to(sid, target_is_directory=True)
    (root / "data" / "fixtures" / "fixture_3asset").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "fixtures" / "fixture_3asset" / "manifest.json", root / "data" / "fixtures" / "fixture_3asset" / "manifest.json")
    return root, sid
