"""Offline round trip of the live path: Yahoo- and ECOS-shaped responses -> adapters -> collector -> immutable snapshot -> ``data validate``.

The responses are hand-made (tests/support/fake_live.py), so this proves that what the adapters emit satisfies every downstream rule
(schema, availability, official sessions, raw/total-return separation, corporate-action reconciliation, FX staleness, calendars), not that
Yahoo's real data does: that is checked only by the data workflow.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from quant_pipeline import collector as C
from quant_pipeline.cli import app
from quant_pipeline.config import Asset, Collector, load_config
from quant_pipeline.live_provider import build_live_provider
from support.fake_live import FakeEcos, FakeYahoo, asset_ids

START, END = "2015-01-02", "2024-12-30"
SETTINGS = Collector(enabled=True, provider="yfinance", cache_root="unused", requests_per_minute=6000, max_concurrency=2, retries=0,
                     backoff_seconds=0.0, fail_on_partial=True)


def records() -> list[dict]:
    out = []
    for sym, ex in asset_ids():
        krx = ex == "XKRX"
        out.append({"asset_id": sym, "exchange": ex, "currency": "KRW" if krx else "USD", "timezone": "Asia/Seoul" if krx else "America/New_York",
                    "calendar": ex, "raw_price_field": "close", "model_price_field": "total_return_index", "lot_size": 1, "quantity_precision": 0,
                    "allow_short": False, "benchmark_eligible": True, "max_weight": 0.30})
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("live")
    (root / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "research_8_12asset.yaml", root / "configs" / "research_8_12asset.yaml")
    universe = root / "universe_source.yaml"
    universe.write_text(yaml.safe_dump({"assets": records()}, sort_keys=False), encoding="utf-8")
    cfg = load_config(root / "configs" / "research_8_12asset.yaml")
    assets = [Asset.model_validate(r) for r in records()]
    yahoo = FakeYahoo(START, END, split={"005930.KS": "2021-09-01"})
    provider = build_live_provider(cfg, assets, yahoo=yahoo, ecos=FakeEcos(3.5))
    out = root / "data" / "snapshots" / "research" / "<snapshot_id>"
    report = C.collect(assets, fx_pair=cfg.fx.pair_id, rates_symbol=cfg.cash.source_id, collector=SETTINGS, provider=provider, start=START, end=END,
                       output=out, cache_root=root / "cache", jitter_seed=1, now=lambda: "2026-01-01T00:00:00Z", universe_source=universe)
    sid = report.snapshot_id.removeprefix("sha256:") if report.snapshot_id else None
    if sid:
        (root / "data" / "snapshots" / "research" / "current").symlink_to(sid, target_is_directory=True)
        (root / "data" / "fixtures" / "fixture_3asset").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "fixtures" / "fixture_3asset" / "manifest.json", root / "data" / "fixtures" / "fixture_3asset" / "manifest.json")
    return {"root": root, "report": report, "provider": provider, "yahoo": yahoo}


def test_the_collector_accepts_the_live_provider_output_and_records_who_supplied_it(world):
    r = world["report"]
    assert r.status == "success", r.issues
    assert r.provider["name"] == "yfinance+ecos" and "Yahoo" in r.provider["license"] and "Bank of Korea" in r.provider["license"]
    snap = world["root"] / "data" / "snapshots" / "research" / "current"
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_kind"] == "research" and "synthetic" not in manifest  # it is not labelled synthetic: the labelling is the caller's duty
    assert {"prices_raw.parquet", "prices_total_return.parquet", "corporate_actions.parquet", "fx_krw_per_usd.parquet", "rates_krw_rfr.parquet"} <= set(manifest["files"])


def test_one_download_per_symbol_serves_prices_total_return_actions_and_metadata(world):
    calls = world["yahoo"].calls
    assert sorted(set(calls)) == sorted([s for s, _ in asset_ids()] + ["KRW=X"])
    assert all(calls.count(s) == 1 for s in set(calls))


def test_the_split_shows_in_the_snapshot_as_a_raw_price_jump_with_a_continuous_total_return_index(world):
    snap = world["root"] / "data" / "snapshots" / "research" / "current"
    raw = pd.read_parquet(snap / "prices_raw.parquet").query("instrument_id == '005930.KS'").set_index("exchange_session").value
    tr = pd.read_parquet(snap / "prices_total_return.parquet").query("instrument_id == '005930.KS'").set_index("exchange_session").value
    day = list(raw.index).index("2021-09-01")
    assert 0.4 < raw.iloc[day] / raw.iloc[day - 1] < 0.6 and abs(tr.iloc[day] / tr.iloc[day - 1] - 1) < 0.08
    acts = pd.read_parquet(snap / "corporate_actions.parquet")
    assert ((acts.instrument_id == "005930.KS") & (acts.action_type == "split")).sum() == 1 and (acts.action_type == "cash_dividend").sum() > 5


def test_data_validate_passes_on_the_live_shaped_snapshot(world):
    cfg = world["root"] / "configs" / "research_8_12asset.yaml"
    r = CliRunner().invoke(app, ["data", "validate", "--config", str(cfg), "--repo-root", str(world["root"])])
    assert r.exit_code == 0 and "data validate: pass" in r.output, r.output


def test_the_quality_report_passes_complete_data_and_names_the_dividend_gap_of_an_incomplete_provider(world):
    from quant_pipeline.data_quality_report import snapshot_report

    snap = world["root"] / "data" / "snapshots" / "research" / "current"
    ok = snapshot_report(snap.resolve(), records(), frozenset({"005930.KS", "SPY"}))
    assert ok["passed"], ok["failures"]
    by = {i["symbol"]: i for i in ok["instruments"]}
    assert by["005930.KS"]["splits"] == 1 and by["005930.KS"]["dividends"] > 20 and by["SPY"]["session_coverage"] > 0.99
    # a provider that omits a payer's dividends (a known weakness of the free service for Korean listings) must fail the report, not pass quietly
    broken = snap.parent / "broken"
    shutil.copytree(snap.resolve(), broken)
    a = pd.read_parquet(broken / "corporate_actions.parquet")
    a[a.instrument_id != "035420.KS"].to_parquet(broken / "corporate_actions.parquet", index=False)
    bad = snapshot_report(broken, records(), frozenset({"035420.KS"}))
    assert not bad["passed"] and any("035420.KS" in f and "dividends a year" in f for f in bad["failures"])
