"""Research universe manifest (spec 4.6, 15; Phase 5 gate: the 8-12 asset manifest constraint). The manifests here are synthetic and
test-only; no research data is committed to the repository."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quant_pipeline.cli import app
from quant_pipeline.config import load_config
from quant_pipeline.data_quality import validate_data, validate_research_universe

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "research_8_12asset.yaml")
FIX = load_config(ROOT / "configs" / "fixture_3asset.yaml")


def record(i: int, exchange: str, eligible: bool = True, max_weight: float = 0.30) -> dict:
    krx = exchange == "XKRX"
    return {"asset_id": f"R{i:02d}", "exchange": exchange, "currency": "KRW" if krx else "USD",
            "timezone": "Asia/Seoul" if krx else "America/New_York", "calendar": exchange, "raw_price_field": "close",
            "model_price_field": "total_return_index", "lot_size": 1, "quantity_precision": 0, "allow_short": False,
            "benchmark_eligible": eligible, "max_weight": max_weight}


def assets(n_krx=5, n_nyse=5, **kw) -> list[dict]:
    return [record(i, "XKRX", **kw) for i in range(n_krx)] + [record(100 + i, "XNYS", **kw) for i in range(n_nyse)]


def write(tmp_path: Path, records, raw_text: str | None = None) -> Path:
    path = tmp_path / CFG.data.universe_manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_text if raw_text is not None else yaml.safe_dump({"assets": records}, sort_keys=False), encoding="utf-8")
    return tmp_path


def failed(results) -> set[str]:
    return {r["check"] for r in results if r["status"] != "pass"}


def test_a_valid_ten_asset_two_market_manifest_passes(tmp_path):
    r = validate_research_universe(CFG, write(tmp_path, assets()))
    assert failed(r) == set() and any(x["check"] == "universe_size_within_bounds" for x in r)
    assert all(x["check"] != "benchmark_feasible[equal_weight_periodic]" or x["status"] == "pass" for x in r)


@pytest.mark.parametrize("n_krx,n_nyse,ok", [(4, 3, False), (4, 4, True), (6, 6, True), (7, 6, False)])
def test_the_risky_asset_count_must_be_within_8_and_12(tmp_path, n_krx, n_nyse, ok):
    r = validate_research_universe(CFG, write(tmp_path, assets(n_krx, n_nyse)))
    assert ("universe_size_within_bounds" not in failed(r)) is ok


def test_both_required_exchanges_and_currencies_must_be_present(tmp_path):
    r = validate_research_universe(CFG, write(tmp_path, assets(10, 0)))
    assert "universe_required_exchanges_and_currencies" in failed(r)
    r2 = validate_research_universe(CFG, write(tmp_path / "b", assets(0, 10)))
    assert "universe_required_exchanges_and_currencies" in failed(r2)


def test_fixture_asset_ids_are_rejected_to_keep_the_roots_separate(tmp_path):
    recs = assets()
    recs[0]["asset_id"] = "KRX_A"
    assert "universe_rejects_fixture_asset_ids" in failed(validate_research_universe(CFG, write(tmp_path, recs)))


def test_duplicate_ids_and_the_cash_asset_as_a_risky_asset_are_rejected(tmp_path):
    dup = assets()
    dup[1]["asset_id"] = dup[0]["asset_id"]
    assert "universe_unique_ids" in failed(validate_research_universe(CFG, write(tmp_path, dup)))
    cash = assets()
    cash[0]["asset_id"] = "CASH_KRW"
    assert "universe_cash_is_not_a_risky_asset" in failed(validate_research_universe(CFG, write(tmp_path / "c", cash)))


def test_a_benchmark_set_that_cannot_be_built_halts_instead_of_dropping_members(tmp_path):
    few = [record(i, "XKRX", eligible=i < 2) for i in range(5)] + [record(100 + i, "XNYS", eligible=False) for i in range(5)]
    assert "benchmark_feasible[equal_weight_periodic]" in failed(validate_research_universe(CFG, write(tmp_path, few)))  # |M| = 2 < 4
    small_caps = validate_research_universe(CFG, write(tmp_path / "s", assets(max_weight=0.05)))
    assert "benchmark_feasible[equal_weight_staged_entry]" in failed(small_caps)  # 10 x 0.05 = 0.5 < B = 0.95


@pytest.mark.parametrize("bad", [
    "assets: [",                                              # malformed YAML
    "assets:\n  - {asset_id: A, asset_id: B}\n",              # duplicate YAML key
    "not_assets: []\n",                                       # wrong root key
    yaml.safe_dump({"assets": [{**record(0, "XKRX"), "extra_field": 1}]}),   # unknown key (extra=forbid)
    yaml.safe_dump({"assets": [{k: v for k, v in record(0, "XKRX").items() if k != "lot_size"}]}),  # missing field
    yaml.safe_dump({"assets": [{**record(0, "XKRX"), "currency": "USD"}]}),  # exchange / currency inconsistent
])
def test_malformed_or_inconsistent_manifests_fail_to_parse(tmp_path, bad):
    r = validate_research_universe(CFG, write(tmp_path, None, raw_text=bad))
    assert failed(r) == {"universe_manifest_parse"}


def test_a_missing_manifest_and_a_fixture_profile_are_reported(tmp_path):
    assert failed(validate_research_universe(CFG, tmp_path)) == {"universe_manifest"}
    assert failed(validate_research_universe(FIX, tmp_path)) == {"universe_manifest"}  # the fixture has no research manifest


def test_data_validate_for_a_snapshot_profile_stops_at_universe_failures(tmp_path):
    assert failed(validate_data(CFG, write(tmp_path, assets(3, 3)))) == {"universe_size_within_bounds"}
    r = validate_data(CFG, write(tmp_path / "ok", assets()))  # universe fine, but no snapshot files exist yet
    assert "files_present" in failed(r) and "universe_size_within_bounds" not in failed(r)


def test_research_and_fixture_roots_are_disjoint():
    fx = {CFG.data.__getattribute__(k) for k in ("manifest", "prices_raw", "prices_total_return", "fx", "rates", "corporate_actions", "initial_holdings")}
    rs = {FIX.data.__getattribute__(k) for k in ("manifest", "prices_raw", "prices_total_return", "fx", "rates", "corporate_actions", "initial_holdings")}
    assert not fx & rs and all("snapshots" in p for p in fx) and all("fixtures" in p for p in rs)
    assert CFG.universe.reject_fixture_asset_ids and CFG.data.collector.enabled is False  # no network in the default research path


def test_the_cli_data_validate_works_for_the_research_profile(tmp_path):
    write(tmp_path, assets())
    r = CliRunner().invoke(app, ["data", "validate", "--config", str(ROOT / "configs" / "research_8_12asset.yaml"), "--repo-root", str(tmp_path)])
    assert r.exit_code == 1 and "ok   universe_size_within_bounds" in r.output and "FAIL files_present" in r.output
