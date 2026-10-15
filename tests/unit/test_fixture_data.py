"""Fixture-v5 data contract tests (spec section 14 scenario list). Evidence class: fixture inputs only."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FX_DIR = ROOT / "data" / "fixtures" / "fixture_3asset"
GEN = ROOT / "tools" / "generate_fixture.py"


def _gen_module():
    spec = importlib.util.spec_from_file_location("generate_fixture", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(FX_DIR / "prices_raw.parquet")


@pytest.fixture(scope="module")
def tr() -> pd.DataFrame:
    return pd.read_parquet(FX_DIR / "prices_total_return.parquet")


def test_manifest_matches_lock_and_files():
    gen = _gen_module()
    manifest = json.loads((FX_DIR / "manifest.json").read_text(encoding="utf-8"))
    lock = gen.verify_lock(FX_DIR / "fixture_generation.lock.json")
    assert manifest["generation_lock_hash"] == gen.jcs_hash(lock)
    assert manifest["generator_source_hash"] == gen.generator_source_hash()
    assert manifest["seed_value"] == 1728 and manifest["seed_stream"] == "fixture_data"
    for name, meta in manifest["files"].items():
        assert gen.sha((FX_DIR / name).read_bytes()) == meta["raw_sha256"], name


def test_regeneration_reproduces_all_logical_hashes(tmp_path):
    out = tmp_path / "regen"
    out.mkdir()
    shutil.copy(FX_DIR / "fixture_generation.lock.json", out / "fixture_generation.lock.json")
    subprocess.run([sys.executable, str(GEN), "--scenario", "fixture-v5", "--output", str(out)], check=True,
                   capture_output=True)
    want = json.loads((FX_DIR / "manifest.json").read_text(encoding="utf-8"))["files"]
    got = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert {k: v["logical_sha256"] for k, v in got.items()} == {k: v["logical_sha256"] for k, v in want.items()}


def test_generator_refuses_overwrite_and_missing_or_stale_lock(tmp_path):
    r = subprocess.run([sys.executable, str(GEN), "--scenario", "fixture-v5", "--output", str(FX_DIR)],
                       check=False, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr
    empty = tmp_path / "nolock"
    r = subprocess.run([sys.executable, str(GEN), "--scenario", "fixture-v5", "--output", str(empty)],
                       check=False, capture_output=True, text=True)
    assert r.returncode != 0 and "missing pre-frozen lock" in r.stderr
    stale = tmp_path / "stale"
    stale.mkdir()
    lock = json.loads((FX_DIR / "fixture_generation.lock.json").read_text(encoding="utf-8"))
    lock["scenario_definition_hash"] = "sha256:" + "0" * 64
    (stale / "fixture_generation.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    r = subprocess.run([sys.executable, str(GEN), "--scenario", "fixture-v5", "--output", str(stale)],
                       check=False, capture_output=True, text=True)
    assert r.returncode != 0 and "lock mismatch" in r.stderr


def test_generator_does_not_import_the_package():
    src = GEN.read_text(encoding="utf-8")
    assert "quant_pipeline" not in src.replace("``quant_pipeline``", "").replace("quant_pipeline`` ", "").split(
        '"""', 2)[2]


def test_geometry_matches_spec(raw):
    xnys = sorted(raw[raw.instrument_id == "NYSE_C"].exchange_session)
    assert sum(s >= "2024-01-02" for s in xnys) == 97
    both = sorted(set(raw[raw.instrument_id == "NYSE_C"].exchange_session)
                  & set(raw[raw.instrument_id == "KRX_A"].exchange_session))
    complete_case = both[1:]
    assert len(complete_case) == 90
    eligible = complete_case[20:]
    assert len(eligible) == 70 and eligible[0] == "2024-01-31" and eligible[-1] == "2024-05-20"
    assert (eligible[40], eligible[49], eligible[50], eligible[59], eligible[60]) == (
        "2024-04-03", "2024-04-17", "2024-04-18", "2024-05-02", "2024-05-03")


def test_one_market_holidays_exist(raw):
    k = set(raw[raw.instrument_id == "KRX_A"].exchange_session)
    n = set(raw[raw.instrument_id == "NYSE_C"].exchange_session)
    assert k - n and n - k  # an XKRX-only session and an XNYS-only session both exist


def test_available_at_is_after_event_and_utc(raw):
    ev = pd.to_datetime(raw.event_timestamp_utc, utc=True)
    av = pd.to_datetime(raw.available_at_utc, utc=True)
    assert ((av - ev) == pd.Timedelta(minutes=15)).all()
    assert raw.event_timestamp_utc.str.endswith("Z").all()
    assert raw.groupby(["instrument_id", "event_timestamp_utc"]).size().max() == 1


def test_known_krx_close_time_from_spec_16_9(raw):
    row = raw[(raw.instrument_id == "KRX_A") & (raw.exchange_session == "2024-01-03")].iloc[0]
    assert row.event_timestamp_utc == "2024-01-03T06:30:00Z"
    nyse = raw[(raw.instrument_id == "NYSE_C") & (raw.exchange_session == "2024-01-03")].iloc[0]
    assert nyse.event_timestamp_utc == "2024-01-03T21:00:00Z"


def test_genuine_unchanged_price_is_distinct_from_missing(raw):
    a = raw[raw.instrument_id == "KRX_A"].sort_values("event_timestamp_utc").reset_index(drop=True)
    i = a.index[a.exchange_session == "2024-02-27"][0]
    assert a.value[i] == a.value[i - 1]
    assert a.value.notna().all()


def test_split_is_neutral_in_total_return_but_not_in_raw(raw, tr):
    b_raw = raw[raw.instrument_id == "KRX_B"].set_index("exchange_session").value
    b_tr = tr[tr.instrument_id == "KRX_B"].set_index("exchange_session").value
    assert b_raw["2024-03-05"] / b_raw["2024-03-04"] < 0.6  # raw halves around a 2:1 split
    assert abs(b_tr["2024-03-05"] / b_tr["2024-03-04"] - 1) < 0.06  # TR has no split jump


def test_dividend_has_separate_ex_and_pay_and_lifts_tr(raw, tr):
    ca = pd.read_parquet(FX_DIR / "corporate_actions.parquet")
    d = ca[ca.action_type == "cash_dividend"].iloc[0]
    assert d.ex_session_date < d.pay_session_date and d.cash_currency == "USD"
    assert pd.to_datetime(d.available_at_utc) < pd.to_datetime(d.ex_timestamp_utc)  # declared before ex-date
    c_raw = raw[raw.instrument_id == "NYSE_C"].set_index("exchange_session").value
    c_tr = tr[tr.instrument_id == "NYSE_C"].set_index("exchange_session").value
    sessions = list(c_raw.index)
    i = sessions.index(d.ex_session_date)
    expected_tr_move = (c_raw.iloc[i] + d.cash_amount) / c_raw.iloc[i - 1]
    assert c_tr.iloc[i] / c_tr.iloc[i - 1] == pytest.approx(expected_tr_move, rel=1e-5)


def test_liquidity_invalidation_is_temporary(raw):
    flagged = raw[raw.quality_flag != "ok"]
    assert set(flagged.instrument_id) == {"NYSE_C"} and len(flagged) == 2


def test_raw_and_total_return_series_differ(raw, tr):
    assert not (raw.value.values == tr.value.values).all()
    assert set(raw.field) == {"close"} and set(tr.field) == {"total_return_index"}


def test_fx_quote_convention_and_staleness(raw):
    fx = pd.read_parquet(FX_DIR / "fx_krw_per_usd.parquet")
    assert set(fx.quote_convention) == {"KRW_per_USD"} and fx.value.between(1000, 1700).all()
    ev = pd.to_datetime(fx.event_timestamp_utc, utc=True).sort_values().reset_index(drop=True)
    # max_fx_stale_hours applies from the first holdings timestamp; the New Year gap before it carries no FX event
    scoped = ev[ev >= pd.Timestamp("2024-01-02T21:15:00Z")]
    gaps = ev.diff()[scoped.index]
    assert gaps.max() <= pd.Timedelta(hours=72)


def test_rates_nonzero_with_publication_step():
    r = pd.read_parquet(FX_DIR / "rates_krw_rfr.parquet")
    assert (r.annual_rate > 0).all() and r.annual_rate.nunique() >= 3
    assert (r.available_at_utc == r.event_timestamp_utc).all()  # 0-business-day publication delay


def test_regime_table_schema_and_blocks_per_spec_7_5():
    full = pd.read_csv(FX_DIR / "fixed_regime_table.csv")
    assert list(full.columns) == ["available_at_utc", "state_0_probability", "state_1_probability", "expected_label"]
    assert len(full) == 97  # one row per XNYS anchor in the decision window
    assert full.available_at_utc.is_monotonic_increasing and full.available_at_utc.is_unique
    full = full.assign(session_date=full.available_at_utc.str[:10])
    krx_days = set(pd.read_parquet(FX_DIR / "prices_raw.parquet").query("instrument_id == 'KRX_A'").exchange_session)
    t = full[full.session_date.isin(krx_days) & (full.session_date >= "2024-01-31")].reset_index(drop=True)
    assert len(t) == 70  # eligible rows
    for lo, hi in [(40, 50), (50, 60), (60, 70)]:  # outer test 1, outer test 2, holdout (0-indexed slices)
        assert set(t.expected_label[lo:hi]) == {"low_risk", "high_risk"}, (lo, hi)
    # no one-day flicker on one-market holidays: exactly one switch per block boundary (7 blocks -> 6 switches)
    assert (full.expected_label != full.expected_label.shift()).iloc[1:].sum() == 6
    assert ((full.state_0_probability + full.state_1_probability - 1).abs() < 1e-9).all()
    # state_0 is the low-risk state: its probability dominates exactly when the label is low_risk (or the 0.5 switch row)
    assert (full[full.expected_label == "low_risk"].state_0_probability >= 0.5).all()
    assert (full[full.expected_label == "high_risk"].state_1_probability >= 0.5).all()


def test_initial_holdings_schema_per_spec_13():
    h = pd.read_csv(FX_DIR / "initial_holdings.csv", dtype=str, keep_default_na=False)
    assert list(h.columns) == ["asset_id", "quantity", "currency", "cash_amount", "as_of_utc"]
    cash = h[h.asset_id == "CASH_KRW"].iloc[0]
    assert cash.quantity == "0" and cash.cash_amount == "1000000000.00" and cash.as_of_utc == "2024-01-02T21:15:00Z"
    risky = h[h.asset_id != "CASH_KRW"]
    assert (risky.cash_amount == "").all() and h.asset_id.is_unique
