#!/usr/bin/env python3
"""Deterministic generator for the ``fixture-v5`` three-asset fixture (spec section 14).

Independent dependency graph: this file MUST NOT import ``quant_pipeline``, run an optimizer, select a trial,
write expected outputs, or touch the network. It reads the seed only from
``reproducibility.random_streams.fixture_data`` of the fixture profile and refuses to run unless a pre-frozen
``fixture_generation.lock.json`` matches this file's hash, the scenario hash, the stream value, and the
calendar-session digest. Accepted command:

    python tools/generate_fixture.py --scenario fixture-v5 --output data/fixtures/fixture_3asset

Provisional file schemas (spec defines only initial_holdings.csv) are recorded as OQ-004 in open-questions.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rfc8785
import yaml

SCENARIO_ID = "fixture-v5"
SEED_STREAM = "fixture_data"
WINDOW_START = "2023-12-28"  # seed observation start session
WINDOW_END = "2024-05-20"
DECISION_START = "2024-01-02"
WARMUP_ROWS = 20
PUBLICATION_DELAY = timedelta(minutes=15)
INITIAL_CASH = "1000000000.00"
INITIAL_AS_OF = "2024-01-02T21:15:00Z"  # XNYS close (21:00Z, EST) + 15 min; equals spec 16.9 cutoff
REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs" / "fixture_3asset.yaml"

# Everything that shapes the generated data lives in SCENARIO; its JCS hash is frozen in the lock.
SCENARIO = {
    "scenario_id": SCENARIO_ID,
    "assets": {
        "KRX_A": {"exchange": "XKRX", "currency": "KRW", "p0": 70000, "drift": 0.0004, "vol": 0.014, "beta": 0.85},
        "KRX_B": {"exchange": "XKRX", "currency": "KRW", "p0": 120000, "drift": 0.0002, "vol": 0.005, "beta": 0.30},
        "NYSE_C": {"exchange": "XNYS", "currency": "USD", "p0": 150.0, "drift": 0.0005, "vol": 0.011, "beta": 0.80},
    },
    "fx": {"p0": 1305.0, "daily_vol": 0.004, "drift": 0.0, "rho_with_factor": 0.0},
    "rates": {
        "base": 0.0350,
        "steps": [["2024-02-15", 0.0340], ["2024-04-11", 0.0325]],
        "event_time_utc": "00:00:00",
    },
    "regime_scale": {"low_risk": 1.0, "high_risk": 2.2},
    # eligible-position blocks (1-indexed, inclusive) -> state; warmup rows are low_risk
    "regime_blocks": [[1, 15, "low_risk"], [16, 28, "high_risk"], [29, 42, "low_risk"], [43, 47, "high_risk"],
                      [48, 55, "low_risk"], [56, 62, "high_risk"], [63, 70, "low_risk"]],
    "unchanged_price": {"asset": "KRX_A", "session": "2024-02-27"},
    "split": {"asset": "KRX_B", "ex_session": "2024-03-05", "num": 2, "den": 1, "declared": "2024-02-20"},
    "cash_dividend": {"asset": "NYSE_C", "ex_session": "2024-02-08", "pay_session": "2024-02-15",
                      "amount_usd": 0.62, "declared": "2024-01-25"},
    "liquidity_invalid": {"asset": "NYSE_C", "sessions": ["2024-03-14", "2024-03-15"]},
    "correlation_design": {"note": "KRX_B low-vol so minimum-variance exceeds max_weight 0.60 (cap-active case); "
                                   "KRX_A/NYSE_C share a high factor loading (Defensive/Diversification overlap case)"},
    "price_rounding": {"KRW": 0, "USD": 2},
    "tr_index_base": 100.0,
    "tr_index_decimals": 6,
}

PRICE_COLUMNS = ["instrument_id", "source_timestamp", "event_timestamp_utc", "available_at_utc", "exchange_session",
                 "field", "value", "currency", "unit", "source_id", "source_revision", "is_final", "is_stale",
                 "quality_flag"]
PRIMARY_KEYS = {
    "prices_raw": ["instrument_id", "event_timestamp_utc"],
    "prices_total_return": ["instrument_id", "event_timestamp_utc"],
    "fx_krw_per_usd": ["pair_id", "event_timestamp_utc"],
    "rates_krw_rfr": ["rate_date"],
    "corporate_actions": ["action_id"],
}


# ---------------------------------------------------------------- hashing (own copy; no package import)
def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_text_hash(raw: bytes) -> str:
    text = raw.decode("utf-8").removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip(" \t") for ln in text.split("\n")]
    return sha(("\n".join(lines).rstrip("\n") + "\n").encode("utf-8"))


def jcs_hash(obj: object) -> str:
    return sha(rfc8785.dumps(obj))


def scenario_hash() -> str:
    return jcs_hash(SCENARIO)


def generator_source_hash() -> str:
    return canonical_text_hash(Path(__file__).read_bytes())


def logical_parquet_hash(path: Path, keys: list[str]) -> str:
    table = pq.read_table(path).sort_by([(k, "ascending") for k in keys])
    table = table.replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    return sha(sink.getvalue().to_pybytes())


def logical_csv_hash(path: Path) -> str:
    return canonical_text_hash(path.read_bytes())


# ---------------------------------------------------------------- calendars
def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def calendar_sessions() -> dict[str, pd.DatetimeIndex]:
    out = {}
    for name in ("XKRX", "XNYS"):
        cal = xc.get_calendar(name)
        out[name] = cal.sessions_in_range(WINDOW_START, WINDOW_END)
    return out


def calendar_digest() -> str:
    cals = {n: xc.get_calendar(n) for n in ("XKRX", "XNYS")}
    payload = {}
    for name, sessions in calendar_sessions().items():
        payload[name] = [[s.strftime("%Y-%m-%d"), _utc(cals[name].session_close(s)).strftime("%Y-%m-%dT%H:%M:%SZ")]
                         for s in sessions]
    return jcs_hash(payload)


def fixture_stream_value() -> int:
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    return int(profile["reproducibility"]["random_streams"][SEED_STREAM])


def expected_lock() -> dict:
    """Content a valid lock must contain (the ``frozen_at_utc`` / ``signoff`` fields are informational)."""
    return {
        "scenario_id": SCENARIO_ID,
        "scenario_definition_hash": scenario_hash(),
        "generator_source_hash": generator_source_hash(),
        "fixture_data_stream": {"name": SEED_STREAM, "value": fixture_stream_value()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "pyarrow": pa.__version__, "exchange_calendars": xc.__version__},
        "calendar_session_digest": calendar_digest(),
        "command": f"tools/generate_fixture.py --scenario {SCENARIO_ID} --output data/fixtures/fixture_3asset",
    }


def verify_lock(lock_path: Path) -> dict:
    if not lock_path.exists():
        raise SystemExit(f"missing pre-frozen lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    want = expected_lock()
    for key, value in want.items():
        if lock.get(key) != value:
            raise SystemExit(f"lock mismatch on {key!r}: lock={lock.get(key)!r} current={value!r}")
    return lock


# ---------------------------------------------------------------- generation
def _eligible_positions(xnys: pd.DatetimeIndex, xkrx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    both = xnys.intersection(xkrx)
    complete_case = both[1:]  # first common session has no preceding observed level
    return list(complete_case[WARMUP_ROWS:])


def _regime_by_session(xnys: pd.DatetimeIndex, eligible: list[pd.Timestamp]) -> dict[pd.Timestamp, str]:
    pos_state = {}
    for lo, hi, label in SCENARIO["regime_blocks"]:
        for p in range(lo, hi + 1):
            pos_state[p] = label
    eligible_state = {session: pos_state[pos] for pos, session in enumerate(eligible, start=1)}
    # An XNYS anchor that is not an eligible row (XKRX closed) inherits the preceding state, so a block never
    # flickers on a one-market holiday. Anchors before the first eligible row (warmup) are low_risk.
    state: dict[pd.Timestamp, str] = {}
    last = "low_risk"
    for s in xnys:
        last = eligible_state.get(s, last)
        state[s] = last
    return state


def _fmt(ts: pd.Timestamp) -> str:
    return _utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_tables(seed: int) -> dict[str, pd.DataFrame]:
    cals = {n: xc.get_calendar(n) for n in ("XKRX", "XNYS")}
    sess = calendar_sessions()
    xnys, xkrx = sess["XNYS"], sess["XKRX"]
    eligible = _eligible_positions(xnys, xkrx)
    regime = _regime_by_session(xnys, eligible)
    dates = sorted(set(xnys) | set(xkrx))
    rng = np.random.Generator(np.random.PCG64(seed))

    # latent log-price paths on the union of session dates (a closed market's moves surface at its next session)
    assets = SCENARIO["assets"]
    names = list(assets)
    latent = {n: np.log(assets[n]["p0"]) for n in names}
    log_fx = float(np.log(SCENARIO["fx"]["p0"]))
    cur_state = "low_risk"
    latent_at: dict[str, dict[pd.Timestamp, float]] = {n: {} for n in names}
    fx_at: dict[pd.Timestamp, float] = {}
    for d in dates:
        cur_state = regime.get(d, cur_state)
        scale = SCENARIO["regime_scale"][cur_state]
        f = rng.standard_normal()
        e = rng.standard_normal(len(names))
        g = rng.standard_normal()
        for i, n in enumerate(names):
            a = assets[n]
            shock = a["beta"] * f + np.sqrt(1.0 - a["beta"] ** 2) * e[i]
            latent[n] += a["drift"] + scale * a["vol"] * shock
            latent_at[n][d] = latent[n]
        log_fx += SCENARIO["fx"]["drift"] + scale * SCENARIO["fx"]["daily_vol"] * g
        fx_at[d] = log_fx

    split = SCENARIO["split"]
    div = SCENARIO["cash_dividend"]
    split_ex = pd.Timestamp(split["ex_session"])
    raw_rows, tr_rows = [], []
    for n in names:
        a = assets[n]
        cal = cals[a["exchange"]]
        decimals = SCENARIO["price_rounding"][a["currency"]]
        prev_raw = prev_adj_raw = None
        tr = SCENARIO["tr_index_base"]
        for s in (xkrx if a["exchange"] == "XKRX" else xnys):
            price = float(np.exp(latent_at[n][s]))
            if n == split["asset"] and s >= split_ex:
                price = price * split["den"] / split["num"]
            raw = round(price, decimals)
            if n == SCENARIO["unchanged_price"]["asset"] and s == pd.Timestamp(SCENARIO["unchanged_price"]["session"]):
                raw = prev_raw  # genuine unchanged observation: a real update with zero return
            flag = "ok"
            if n == SCENARIO["liquidity_invalid"]["asset"] and s.strftime("%Y-%m-%d") in SCENARIO["liquidity_invalid"]["sessions"]:
                flag = "liquidity_invalid"
            adj_raw = raw * (split["num"] / split["den"] if n == split["asset"] and s >= split_ex else 1.0)
            if prev_raw is not None:
                dividend = div["amount_usd"] if n == div["asset"] and s == pd.Timestamp(div["ex_session"]) else 0.0
                tr = tr * (adj_raw + dividend) / prev_adj_raw
            prev_raw, prev_adj_raw = raw, adj_raw
            close_utc = _utc(cal.session_close(s))
            base = {
                "instrument_id": n,
                "source_timestamp": close_utc.tz_convert(
                    "Asia/Seoul" if a["exchange"] == "XKRX" else "America/New_York").isoformat(),
                "event_timestamp_utc": _fmt(close_utc),
                "available_at_utc": _fmt(close_utc + PUBLICATION_DELAY),
                "exchange_session": s.strftime("%Y-%m-%d"),
                "currency": a["currency"],
                "source_id": "fixture-v5",
                "source_revision": "1",
                "is_final": True,
                "is_stale": False,
                "quality_flag": flag,
            }
            raw_rows.append({**base, "field": "close", "value": float(raw), "unit": a["currency"] + "/share"})
            tr_rows.append({**base, "field": "total_return_index", "value": round(tr, SCENARIO["tr_index_decimals"]),
                            "unit": "index"})

    fx_rows = []
    for s in dates:
        for cal_name in ("XKRX", "XNYS"):
            if s in (xkrx if cal_name == "XKRX" else xnys):
                t = _utc(cals[cal_name].session_close(s))
                fx_rows.append({"pair_id": "USDKRW", "event_timestamp_utc": _fmt(t),
                                "available_at_utc": _fmt(t + PUBLICATION_DELAY),
                                "value": round(float(np.exp(fx_at[s])), 2), "quote_convention": "KRW_per_USD",
                                "source_id": "fixture-v5"})

    rate_rows = []
    steps = {pd.Timestamp(d): r for d, r in SCENARIO["rates"]["steps"]}
    rate = SCENARIO["rates"]["base"]
    for s in xkrx:
        rate = steps.get(s, rate)
        t = pd.Timestamp(f"{s.strftime('%Y-%m-%d')}T{SCENARIO['rates']['event_time_utc']}", tz="UTC")
        rate_rows.append({"rate_date": s.strftime("%Y-%m-%d"), "event_timestamp_utc": _fmt(t),
                          "available_at_utc": _fmt(t), "annual_rate": rate, "quote_basis": "annual_simple",
                          "day_count": "ACT/365F", "source_id": "fixture_krw_rfr"})

    def _close(cal_name: str, day: str) -> pd.Timestamp:
        return _utc(cals[cal_name].session_close(pd.Timestamp(day)))

    ca_rows = [
        {"action_id": "CA-0001", "instrument_id": split["asset"], "action_type": "split",
         "declared_at_utc": f"{split['declared']}T00:00:00Z", "available_at_utc": f"{split['declared']}T00:00:00Z",
         "ex_session_date": split["ex_session"], "ex_timestamp_utc": _fmt(_close("XKRX", split["ex_session"])),
         "pay_session_date": None, "pay_timestamp_utc": None, "ratio_num": split["num"], "ratio_den": split["den"],
         "cash_amount": None, "cash_currency": None},
        {"action_id": "CA-0002", "instrument_id": div["asset"], "action_type": "cash_dividend",
         "declared_at_utc": f"{div['declared']}T00:00:00Z", "available_at_utc": f"{div['declared']}T00:00:00Z",
         "ex_session_date": div["ex_session"], "ex_timestamp_utc": _fmt(_close("XNYS", div["ex_session"])),
         "pay_session_date": div["pay_session"], "pay_timestamp_utc": _fmt(_close("XNYS", div["pay_session"])),
         "ratio_num": None, "ratio_den": None, "cash_amount": div["amount_usd"], "cash_currency": "USD"},
    ]

    # fixed regime table: one row per XNYS anchor in the decision window, keyed by decision cutoff
    dec_start = pd.Timestamp(DECISION_START)
    reg_rows, prev_label = [], None
    for s in xnys[xnys >= dec_start]:
        label = regime[s]
        p_high = 0.9 if label == "high_risk" else 0.1
        if prev_label is not None and label != prev_label:
            p_high = 0.5  # transition row, so the table is not purely binary
        prev_label = label
        cutoff = _utc(cals["XNYS"].session_close(s)) + PUBLICATION_DELAY
        # schema fixed by spec 7.5; state_0 is the lower-volatility state (low_risk), state_1 the higher (high_risk)
        reg_rows.append({"available_at_utc": _fmt(cutoff), "state_0_probability": round(1.0 - p_high, 6),
                         "state_1_probability": p_high, "expected_label": label})

    holdings = [{"asset_id": "CASH_KRW", "quantity": 0, "currency": "KRW", "cash_amount": INITIAL_CASH,
                 "as_of_utc": INITIAL_AS_OF}]
    for n in names:
        holdings.append({"asset_id": n, "quantity": 0, "currency": assets[n]["currency"], "cash_amount": "",
                         "as_of_utc": INITIAL_AS_OF})

    return {
        "prices_raw": pd.DataFrame(raw_rows, columns=PRICE_COLUMNS),
        "prices_total_return": pd.DataFrame(tr_rows, columns=PRICE_COLUMNS),
        "fx_krw_per_usd": pd.DataFrame(fx_rows),
        "rates_krw_rfr": pd.DataFrame(rate_rows),
        "corporate_actions": pd.DataFrame(ca_rows),
        "fixed_regime_table": pd.DataFrame(reg_rows),
        "initial_holdings": pd.DataFrame(holdings),
    }


def write_tables(tables: dict[str, pd.DataFrame], out: Path) -> dict[str, dict]:
    files: dict[str, dict] = {}
    for name, df in tables.items():
        if name in PRIMARY_KEYS:
            path = out / f"{name}.parquet"
            df = df.sort_values(PRIMARY_KEYS[name], kind="mergesort").reset_index(drop=True)
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
            logical = logical_parquet_hash(path, PRIMARY_KEYS[name])
        else:
            path = out / f"{name}.csv"
            df.to_csv(path, index=False, lineterminator="\n")
            logical = logical_csv_hash(path)
        files[path.name] = {"raw_sha256": sha(path.read_bytes()), "logical_sha256": logical, "rows": len(df)}
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.scenario != SCENARIO_ID:
        raise SystemExit(f"unknown scenario {args.scenario!r}")
    out = args.output
    if (out / "manifest.json").exists():
        raise SystemExit(f"refusing to overwrite immutable fixture: {out / 'manifest.json'} exists")
    lock_path = out / "fixture_generation.lock.json"
    lock = verify_lock(lock_path)

    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="fixture_gen_", dir=out.parent))
    try:
        files = write_tables(build_tables(fixture_stream_value()), tmp)
        manifest = {
            "manifest_version": 1,
            "scenario_id": SCENARIO_ID,
            "generation_lock_hash": jcs_hash(lock),
            "generator_source_hash": generator_source_hash(),
            "scenario_definition_hash": scenario_hash(),
            "seed_stream": SEED_STREAM,
            "seed_value": fixture_stream_value(),
            "calendar_session_digest": calendar_digest(),
            "files": dict(sorted(files.items())),
        }
        for fname in files:
            shutil.move(str(tmp / fname), str(out / fname))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                           newline="\n")
        shutil.move(str(tmp / "manifest.json"), str(out / "manifest.json"))  # manifest is written last
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote fixture to {out} at {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
