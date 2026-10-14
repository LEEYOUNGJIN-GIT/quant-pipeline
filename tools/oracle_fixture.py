#!/usr/bin/env python3
"""Independent deterministic-kernel oracle for the fixture (spec 16.11).

Independent means: this file MUST NOT import ``quant_pipeline``, call its CLI, read run outputs, invoke
CVXPY/OSQP, fit production models, select a trial, or share production helper functions. It reads only the
immutable fixture inputs, the frozen profile and the SciPy Sobol engine named in the lock.

Kernels: calendar/row ownership; daily+weekly as-of panel; event-queue ordering; exact Decimal ledger replay of a
scripted fill set; HMM forward filter from supplied parameters (exact Fractions for the section 16.6 case);
Sobol coordinate->index mapping; stationary-bootstrap index generation.

Regenerating oracle files is an explicit, non-verification command (``--write``). Verification and tests never write.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import rfc8785
import yaml
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "fixture_3asset"
PROFILE = ROOT / "configs" / "fixture_3asset.yaml"
OUT = ROOT / "tests" / "expected" / "fixture_3asset" / "oracle"
DELAY = pd.Timedelta(minutes=15)
ALGORITHMS = {
    "calendar": "oracle-calendar-v1",
    "panel": "oracle-asof-panel-v1",
    "events": "oracle-event-queue-v1",
    "ledger": "oracle-decimal-ledger-v1",
    "hmm": "oracle-forward-filter-v1",
    "sobol": "oracle-sobol-floor-map-v1",
    "bootstrap": "oracle-stationary-bootstrap-v1-politis-romano",
}
PRIORITY = {"corporate_action": 0, "cash_accrual": 1, "sell_fill": 2, "fx_conversion": 3, "buy_fill": 4,
            "valuation": 5}
EXCH_RANK = {"XKRX": 0, "XNYS": 1, "": 2}
ASSETS = {"KRX_A": ("XKRX", "KRW"), "KRX_B": ("XKRX", "KRW"), "NYSE_C": ("XNYS", "USD")}
D = Decimal


def sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def z(ts) -> str:
    return utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def profile() -> dict:
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ calendar and row ownership
def calendars():
    fw = profile()["data"]["fixture_window"]
    start, end = fw["seed_observation_start_session"], fw["decision_end_session"]
    cals = {n: xc.get_calendar(n) for n in ("XKRX", "XNYS")}
    return cals, {n: cals[n].sessions_in_range(start, end) for n in cals}, fw


def calendar_rows():
    cals, sess, fw = calendars()
    both = sess["XNYS"].intersection(sess["XKRX"])
    complete_case = list(both[1:])  # the first common session has no preceding observed level
    warm = fw["warmup_rows"]
    eligible = complete_case[warm:]
    position = {s: i for i, s in enumerate(eligible, start=1)}
    ev = profile()["evaluation"]
    otr, ots, fh = ev["outer"]["initial_train_rows"], ev["outer"]["test_rows"], profile()["final_holdout"]["rows"]
    bounds = {"outer1": (otr + 1, otr + ots), "outer2": (otr + ots + 1, otr + 2 * ots),
              "holdout": (len(eligible) - fh + 1, len(eligible))}
    owner_sessions = {k: (eligible[lo - 1], eligible[hi - 1]) for k, (lo, hi) in bounds.items()}
    rows = []
    for s in sess["XNYS"]:
        owner = next((k for k, (a, b) in owner_sessions.items() if a <= s <= b), "")
        rows.append({
            "calendar": "XNYS", "session_date": s.strftime("%Y-%m-%d"), "close_utc": z(cals["XNYS"].session_close(s)),
            "is_anchor": s >= pd.Timestamp(fw["decision_start_session"]),
            "also_xkrx_open": s in sess["XKRX"], "complete_case": s in set(complete_case),
            "eligible_position": position.get(s, 0),
            "role": ("warmup" if s in set(complete_case[:warm]) else "eligible" if s in position else "none"),
            "valuation_owner": owner if s >= pd.Timestamp(fw["decision_start_session"]) else "",
        })
    for s in sess["XKRX"]:
        rows.append({"calendar": "XKRX", "session_date": s.strftime("%Y-%m-%d"),
                     "close_utc": z(cals["XKRX"].session_close(s)), "is_anchor": False,
                     "also_xkrx_open": True, "complete_case": False, "eligible_position": 0, "role": "none",
                     "valuation_owner": ""})
    return rows, eligible, owner_sessions


# ------------------------------------------------------------------ panel (daily + weekly)
def load_levels() -> dict[str, pd.DataFrame]:
    tr = pd.read_parquet(FIXTURE / "prices_total_return.parquet")
    out = {}
    for a, g in tr.groupby("instrument_id"):
        g = g.assign(ev=pd.to_datetime(g.event_timestamp_utc, utc=True), av=pd.to_datetime(g.available_at_utc, utc=True))
        out[a] = g.sort_values("ev").reset_index(drop=True)
    return out


def asof(levels: pd.DataFrame, cutoff: pd.Timestamp):
    ok = levels[levels.av <= cutoff]
    return None if ok.empty else ok.iloc[-1]


def panel_rows():
    cals, sess, _fw = calendars()
    levels = load_levels()
    rows = []
    prev: dict[str, pd.Series | None] = {a: None for a in ASSETS}
    for s in sess["XNYS"]:
        cutoff = utc(cals["XNYS"].session_close(s)) + DELAY
        row = {"kind": "daily", "cutoff_utc": z(cutoff), "session_date": s.strftime("%Y-%m-%d")}
        all_updated = True
        for a in ASSETS:
            cur = asof(levels[a], cutoff)
            updated = cur is not None and (prev[a] is None or cur.ev > prev[a].ev)
            ret = (cur.value / prev[a].value - 1.0) if updated and prev[a] is not None else None
            row[f"{a}_level"] = None if cur is None else float(cur.value)
            row[f"{a}_updated"] = bool(updated)
            row[f"{a}_return"] = ret
            row[f"{a}_age_hours"] = None if cur is None else (cutoff - cur.ev).total_seconds() / 3600
            all_updated &= bool(updated) and ret is not None
            if updated:
                prev[a] = cur
        row["complete_case"] = bool(all_updated)
        rows.append(row)
    # weekly Friday 23:59:59Z common cutoff
    first, last = sess["XNYS"][0], sess["XNYS"][-1]
    fridays = pd.date_range(first - pd.Timedelta(days=first.weekday()) + pd.Timedelta(days=4), last, freq="7D")
    prevw: dict[str, pd.Series | None] = {a: None for a in ASSETS}
    for f in fridays:
        cutoff = utc(f.strftime("%Y-%m-%d") + "T23:59:59")
        row = {"kind": "weekly", "cutoff_utc": z(cutoff), "session_date": f.strftime("%Y-%m-%d")}
        all_ok = True
        for a in ASSETS:
            cur = asof(levels[a], cutoff)
            fresh = cur is not None and (cutoff - cur.ev) <= pd.Timedelta(days=7)
            updated = fresh and (prevw[a] is None or cur.ev > prevw[a].ev)
            ret = (cur.value / prevw[a].value - 1.0) if updated and prevw[a] is not None else None
            row[f"{a}_level"] = None if cur is None else float(cur.value)
            row[f"{a}_updated"] = bool(updated)
            row[f"{a}_return"] = ret
            row[f"{a}_age_hours"] = None if cur is None else (cutoff - cur.ev).total_seconds() / 3600
            all_ok &= ret is not None
            if cur is not None:
                prevw[a] = cur
        row["complete_case"] = bool(all_ok)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ event queue and ledger
def next_close(cals, exchange: str, cutoff: pd.Timestamp):
    start = cutoff.tz_convert("UTC").tz_localize(None).normalize()
    for s in cals[exchange].sessions_in_range(start, start + pd.Timedelta(days=10)):
        close = utc(cals[exchange].session_close(s))
        if close > cutoff:
            return s, close
    raise RuntimeError("no executable close found")


def base_events(include_slots: bool):
    cals, sess, fw = calendars()
    _, eligible, _ = calendar_rows()
    ca = pd.read_parquet(FIXTURE / "corporate_actions.parquet")
    rates = pd.read_parquet(FIXTURE / "rates_krw_rfr.parquet")
    ev = []

    def add(ts, kind, exch="", asset="", order="", **extra):
        ev.append({"ts": utc(ts), "kind": kind, "exchange": exch, "asset_id": asset, "order_id": order, **extra})

    for r in ca.itertuples():
        exch = ASSETS[r.instrument_id][0]
        add(r.ex_timestamp_utc, "corporate_action", exch, r.instrument_id, f"{r.action_id}-ex",
            action_type=r.action_type, phase="ex", num=r.ratio_num, den=r.ratio_den, cash_amount=r.cash_amount)
        if r.action_type == "cash_dividend":
            add(r.pay_timestamp_utc, "corporate_action", exch, r.instrument_id, f"{r.action_id}-pay",
                action_type=r.action_type, phase="pay", num=None, den=None, cash_amount=r.cash_amount)
    for r in rates.itertuples():
        add(r.available_at_utc, "cash_accrual", "", "CASH_KRW", f"RATE-{r.rate_date}", annual_rate=r.annual_rate)
    for s in sess["XNYS"][sess["XNYS"] >= pd.Timestamp(fw["decision_start_session"])]:
        t = utc(cals["XNYS"].session_close(s)) + DELAY
        add(t, "cash_accrual", "", "CASH_KRW", f"VAL-ACCR-{s:%Y-%m-%d}", annual_rate=None)
        add(t, "valuation", "", "PORTFOLIO", f"VAL-{s:%Y-%m-%d}")
    if include_slots:
        for pos, s in enumerate(eligible, start=1):
            cutoff = utc(cals["XNYS"].session_close(s)) + DELAY
            for a, (exch, _) in ASSETS.items():
                _, close = next_close(cals, exch, cutoff)
                for side in ("sell", "buy"):
                    add(close, f"{side}_fill", exch, a, f"D{pos:03d}-{a}-{side}", decision=pos)
    return ev


def sort_events(ev):
    ev = sorted(ev, key=lambda e: (e["ts"], PRIORITY[e["kind"]], EXCH_RANK[e["exchange"]], e["asset_id"], e["order_id"]))
    for i, e in enumerate(ev, start=1):
        e["event_id"] = f"E{i:05d}"
    return ev


# scripted fills used only to exercise ledger arithmetic (not a strategy)
SCRIPTED = [("2024-02-01", "KRX_A", 3000), ("2024-02-01", "KRX_B", 1500), ("2024-02-01", "NYSE_C", 2000),
            ("2024-03-20", "KRX_A", -1000), ("2024-03-20", "NYSE_C", -500), ("2024-04-17", "KRX_B", 1000)]


def q2(x: Decimal) -> Decimal:
    return x.quantize(D("0.01"), rounding=ROUND_HALF_EVEN)


def ledger_projection():
    cals, _sess, _fw = calendars()
    prof = profile()
    costs = prof["costs"]
    comm_bps, spread_bps = D(str(costs["commission_bps"])), D(str(costs["half_spread_slippage_bps"]))
    fx_bps = D(str(costs["fx_conversion_bps"]))
    raw = pd.read_parquet(FIXTURE / "prices_raw.parquet")
    raw = raw.assign(ev=pd.to_datetime(raw.event_timestamp_utc, utc=True), av=pd.to_datetime(raw.available_at_utc, utc=True))
    px = {a: g.sort_values("ev").reset_index(drop=True) for a, g in raw.groupby("instrument_id")}
    fxq = pd.read_parquet(FIXTURE / "fx_krw_per_usd.parquet")
    fxq = fxq.assign(av=pd.to_datetime(fxq.available_at_utc, utc=True)).sort_values("av").reset_index(drop=True)

    ev = base_events(include_slots=False)
    for d, a, qty in SCRIPTED:
        exch = ASSETS[a][0]
        t = utc(cals[exch].session_close(pd.Timestamp(d)))
        add_kind = "buy_fill" if qty > 0 else "sell_fill"
        ev.append({"ts": t, "kind": add_kind, "exchange": exch, "asset_id": a, "order_id": f"S-{d}-{a}", "qty": qty})
    ev = sort_events(ev)

    cash = D(str(prof_cash(FIXTURE)))
    qty = {a: 0 for a in ASSETS}
    rate = D(0)
    last_ts = utc(profile_initial_asof())
    costs_cum = D(0)
    entitled: dict[str, int] = {}
    out = []

    def latest_px(a, ts):
        g = px[a]
        ok = g[g.av <= ts]
        return D(str(ok.iloc[-1].value)) if not ok.empty else None

    def latest_fx(ts):
        ok = fxq[fxq.av <= ts]
        return D(str(ok.iloc[-1].value)) if not ok.empty else None

    for e in ev:
        if e["ts"] < last_ts:
            # before the opening holdings timestamp only the rate in force is updated; nothing accrues or posts
            if e["kind"] == "cash_accrual" and e.get("annual_rate") is not None:
                rate = D(str(e["annual_rate"]))
            continue
        # accrue cash since the previous ledger event at the rate in force after that event
        if e["ts"] > last_ts and cash > 0 and rate > 0:
            days = D((e["ts"].normalize() - last_ts.normalize()).days)  # ACT days between calendar dates
            cash = q2(cash * (1 + rate * days / D(365)))
        last_ts = e["ts"]
        note = ""
        k = e["kind"]
        if k == "cash_accrual" and e.get("annual_rate") is not None:
            rate = D(str(e["annual_rate"]))
        elif k == "corporate_action":
            if e["phase"] == "ex":
                entitled[e["order_id"].removesuffix("-ex")] = qty[e["asset_id"]]  # ex-date holders are paid
                if e["action_type"] == "split":
                    qty[e["asset_id"]] = qty[e["asset_id"]] * int(e["num"]) // int(e["den"])
                    note = "split_quantity_adjust"
            elif e["phase"] == "pay":
                f = latest_fx(e["ts"])
                held = entitled[e["order_id"].removesuffix("-pay")]
                cash = q2(cash + D(held) * D(str(e["cash_amount"])) * f)
                note = f"dividend_credit_fx={f}"
        elif k in ("buy_fill", "sell_fill") and "qty" in e:
            a, n = e["asset_id"], e["qty"]
            price = latest_px_at_close(px[a], e["ts"])
            fx = latest_fx(e["ts"]) if ASSETS[a][1] == "USD" else None
            gross = q2(abs(D(n)) * price * (fx if fx else D(1)))
            # each cost component is an explicit posting (spec 10.1) and is quantized on its own
            cost = (q2(gross * comm_bps / D(10000)) + q2(gross * spread_bps / D(10000))
                    + (q2(gross * fx_bps / D(10000)) if fx else D(0)))
            cash = cash - gross * (1 if n > 0 else -1) - cost
            if cash < 0 or qty[a] + n < 0:
                raise RuntimeError(f"scripted fill infeasible at {e['order_id']}")
            qty[a] += n
            costs_cum += cost
            note = f"px={price};fx={fx};gross={gross};cost={cost}"
        nav = None
        if k == "valuation":
            total = cash
            fx = latest_fx(e["ts"])
            for a, n in qty.items():
                if n:
                    m = latest_px(a, e["ts"])
                    total += D(n) * m * (fx if ASSETS[a][1] == "USD" else D(1))
            nav = q2(total)
        out.append({"event_id": e["event_id"], "ts": z(e["ts"]), "kind": k, "order_id": e["order_id"],
                    "cash_krw": str(cash), "qty_KRX_A": qty["KRX_A"], "qty_KRX_B": qty["KRX_B"],
                    "qty_NYSE_C": qty["NYSE_C"], "cumulative_costs": str(costs_cum),
                    "nav_krw": "" if nav is None else str(nav), "note": note})
    return out


def latest_px_at_close(g: pd.DataFrame, ts: pd.Timestamp) -> Decimal:
    ok = g[g.ev == ts]
    if ok.empty:
        raise RuntimeError(f"no close at {ts}")
    return D(str(ok.iloc[-1].value))


def prof_cash(fixture: Path) -> str:
    rows = list(csv.DictReader((fixture / "initial_holdings.csv").open(encoding="utf-8")))
    return next(r["cash_amount"] for r in rows if r["asset_id"] == "CASH_KRW")


def profile_initial_asof() -> str:
    rows = list(csv.DictReader((FIXTURE / "initial_holdings.csv").open(encoding="utf-8")))
    return rows[0]["as_of_utc"]


# ------------------------------------------------------------------ HMM forward filter (supplied parameters only)
def filter_fractions(start, trans, liks):
    """Exact rational forward filter; the first row uses ``start`` as its predicted prior."""
    out, pred = [], [Fraction(x) for x in start]
    for i, lik in enumerate(liks):
        if i > 0:
            prev = out[-1]
            pred = [sum(prev[r] * Fraction(trans[r][c]) for r in range(len(prev))) for c in range(len(prev))]
        joint = [p * Fraction(l) for p, l in zip(pred, lik, strict=True)]
        tot = sum(joint)
        out.append([j / tot for j in joint])
    return out


def gaussian_diag(x, mean, var):
    return math.exp(-0.5 * sum((xi - m) ** 2 / v + math.log(2 * math.pi * v) for xi, m, v in zip(x, mean, var, strict=True)))


def hmm_golden():
    trans = [["4/5", "1/5"], ["1/10", "9/10"]]
    liks = [["3/5", "2/5"], ["3/10", "7/10"], ["9/10", "1/5"]]
    post = filter_fractions(["1/2", "1/2"], trans, liks)
    boundary = post[-1]
    oos = [sum(boundary[r] * Fraction(trans[r][c]) for r in range(2)) for c in range(2)]
    # float Gaussian emission case, supplied parameters
    means, variances = [[0.0, 0.0], [1.0, -1.0]], [[1.0, 1.0], [0.5, 0.5]]
    obs = [[0.1, -0.2], [0.9, -1.1], [1.2, -0.7], [0.0, 0.3]]
    T = [[0.95, 0.05], [0.10, 0.90]]
    pred, filt = [0.6, 0.4], []
    for i, o in enumerate(obs):
        if i > 0:
            pred = [sum(filt[-1][r] * T[r][c] for r in range(2)) for c in range(2)]
        j = [pred[s] * gaussian_diag(o, means[s], variances[s]) for s in range(2)]
        filt.append([x / sum(j) for x in j])
    return {
        "algorithm": ALGORITHMS["hmm"],
        "rational_case": {
            "start": ["1/2", "1/2"], "transition": trans, "likelihood_rows": liks,
            "posteriors": [[str(x) for x in row] for row in post], "boundary": [str(x) for x in boundary],
            "first_oos_predicted": [str(x) for x in oos], "stationary": ["1/3", "2/3"],
        },
        "gaussian_case": {"start": [0.6, 0.4], "transition": T, "means": means, "variances": variances,
                          "observations": obs, "filtered": filt},
    }


# ------------------------------------------------------------------ Sobol mapping and bootstrap indices
def sobol_rows():
    prof = profile()
    dims = {
        "agents.defensive.overlay_gamma_candidates": prof["agents"]["defensive"]["overlay_gamma_candidates"],
        "objective.lambda_cost_candidates": prof["objective"]["lambda_cost_candidates"],
        "objective.lambda_track_candidates": prof["objective"]["lambda_track_candidates"],
    }
    # a trial's resolved parameters include the single-valued candidate dimensions too (OQ-024)
    singles = {
        "agents.mean_reversion.entry_z_candidates": prof["agents"]["mean_reversion"]["entry_z_candidates"][0],
        "agents.trend.horizon_weight_candidates": prof["agents"]["trend"]["horizon_weight_candidates"][0],
        "objective.lambda_risk_candidates": prof["objective"]["lambda_risk_candidates"][0],
    }
    names = sorted(dims)  # canonical parameter-name order; candidate values keep declared order
    sizes = [len(dict.fromkeys(map(json.dumps, dims[n]))) for n in names]
    cardinality = math.prod(sizes)
    cap = prof["evaluation"]["max_parameter_trials"]
    target = min(cap, cardinality)
    max_points = max(1024, 32 * cardinality)
    seed = prof["reproducibility"]["random_streams"]["trial_sobol"]
    engine = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    seen, rows = {}, []
    for idx in range(max_points):
        u = engine.random(1)[0]
        mapped = [min(math.floor(u[j] * sizes[j]), sizes[j] - 1) for j in range(len(names))]
        params = {**singles, **{n: dims[n][m] for n, m in zip(names, mapped, strict=True)}}
        trial_hash = sha(rfc8785.dumps(params))
        dup = trial_hash in seen
        rows.append({"point_index": idx, **{f"u_{n}": repr(float(u[j])) for j, n in enumerate(names)},
                     **{f"idx_{n}": m for n, m in zip(names, mapped, strict=True)}, "trial_hash": trial_hash,
                     "duplicate_of_point": seen.get(trial_hash, ""), "counts_toward_cap": not dup})
        if not dup:
            seen[trial_hash] = idx
        if len(seen) >= target:
            break
    else:
        raise RuntimeError("sampler failure: distinct count not reached")
    stop = "cap_reached" if len(seen) >= cap and cap < cardinality else "candidate_space_exhausted"
    if len(seen) == target and cap == target and cap == cardinality:
        stop = "cap_reached"
    return rows, {"dimension_order": names, "sizes": sizes, "cardinality": cardinality, "target": target,
                  "seed_stream": "trial_sobol", "seed": seed, "stop_reason": stop,
                  "scipy": __import__("scipy").__version__, "algorithm": ALGORITHMS["sobol"]}


def bootstrap_matrix():
    b = profile()["bootstrap"]
    n, reps, length = b["expected_oos_rows"], b["replicates"], b["mean_block_length"]
    rng = np.random.Generator(np.random.PCG64(profile()["reproducibility"]["random_streams"]["bootstrap"]))
    p = 1.0 / length
    m = np.empty((reps, n), dtype="<i8")
    for r in range(reps):
        m[r, 0] = rng.integers(0, n)
        for t in range(1, n):
            m[r, t] = rng.integers(0, n) if rng.random() < p else (m[r, t - 1] + 1) % n
    return m


# ------------------------------------------------------------------ writing
def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def build_all() -> dict[str, bytes | str]:
    """Return {filename: content} so determinism can be checked without touching disk."""
    files: dict[str, bytes] = {}
    cal, _, _ = calendar_rows()
    files["calendar_sessions.csv"] = _csv_bytes(cal)
    pan = pd.DataFrame(panel_rows())
    files["panel_daily_weekly.csv"] = pan.to_csv(index=False, lineterminator="\n").encode()
    ev = sort_events(base_events(include_slots=True))
    files["event_queue.csv"] = _csv_bytes([{
        "event_id": e["event_id"], "ts": z(e["ts"]), "kind": e["kind"], "priority": PRIORITY[e["kind"]],
        "exchange": e["exchange"], "asset_id": e["asset_id"], "order_id": e["order_id"]} for e in ev])
    files["ledger_projection.csv"] = _csv_bytes(ledger_projection())
    files["hmm_filter.json"] = (json.dumps(hmm_golden(), indent=2, sort_keys=True) + "\n").encode()
    srows, smeta = sobol_rows()
    files["sobol_points.csv"] = _csv_bytes(srows)
    files["sobol_meta.json"] = (json.dumps(smeta, indent=2, sort_keys=True) + "\n").encode()
    m = bootstrap_matrix()
    files["bootstrap_indices.sha256"] = (sha(m.tobytes(order="C")) + "\n").encode()
    return files


def _csv_bytes(rows: list[dict]) -> bytes:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def oracle_manifest(files: dict[str, bytes]) -> dict:
    inp = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    prof = profile()
    return {
        "oracle_source_hash": sha(Path(__file__).read_bytes()),
        "input_manifest_hash": sha(rfc8785.dumps(inp)),
        "algorithm_ids": ALGORITHMS,
        "seed_streams": {k: prof["reproducibility"]["random_streams"][k] for k in ("trial_sobol", "bootstrap")},
        "bootstrap_shape": [prof["bootstrap"]["replicates"], prof["bootstrap"]["expected_oos_rows"]],
        "output_hashes": {k: sha(v) for k, v in sorted(files.items())},
        "evidence_class": "independent_oracle",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="explicit non-verification regeneration of oracle files")
    args = ap.parse_args()
    files = build_all()
    manifest = oracle_manifest(files)
    if not args.write:
        print(json.dumps(manifest["output_hashes"], indent=2))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} is not empty; oracle files are immutable. Review and remove deliberately to update.")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (OUT / name).write_bytes(content)
    (OUT / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                              newline="\n")
    print(f"wrote {len(files) + 1} oracle files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
