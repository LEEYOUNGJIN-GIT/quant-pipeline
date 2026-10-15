"""Data-quality gates (spec 4.5). Hard failures halt; each check reports pass/fail with a concrete detail."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from .calendars import Calendars, fmt, parse_delay
from .config import Config
from .data import load_initial_holdings

Result = dict[str, object]


def _r(name: str, ok: bool, detail: str) -> Result:
    return {"check": name, "status": "pass" if ok else "fail", "detail": detail}


def validate_data(cfg: Config, root: Path) -> list[Result]:
    out: list[Result] = []
    d = cfg.data
    if d.mode == "snapshot":
        out.extend(validate_research_universe(cfg, root))
        if any(r["status"] != "pass" for r in out):
            return out
    files = {"prices_raw": d.prices_raw, "prices_total_return": d.prices_total_return, "fx": d.fx, "rates": d.rates,
             "corporate_actions": d.corporate_actions, "initial_holdings": d.initial_holdings, "manifest": d.manifest}
    absent = [f"{k}={v}" for k, v in files.items() if not (root / v).exists()]
    out.append(_r("files_present", not absent, "all present" if not absent else f"missing {absent}"))
    if absent:
        return out
    if d.mode == "snapshot":  # spec 4.6: the pointer must resolve to the content-addressed, fully verified snapshot before anything is read
        from .snapshot import SnapshotResolutionError, resolve_snapshot, scan_contamination

        try:
            snap = resolve_snapshot(cfg, root)
        except SnapshotResolutionError as e:
            out.append(_r("snapshot_resolved", False, str(e)))
            return out
        out.append(_r("snapshot_resolved", True, f"{snap.snapshot_id} at {snap.directory.name}"))
        scan = scan_contamination(cfg, root, snap)
        out.extend(_r(name, ok, detail) for name, ok, detail in scan)
        if not all(ok for _, ok, _ in scan):
            return out
        from .snapshot import hydrate_universe

        cfg = hydrate_universe(cfg, snap)  # the records the rest of the checks (and later the engine) use are the verified, hashed ones
        from .calendars import build_row_map, window_from_manifest
        from .snapshot import complete_case_coverage

        try:
            rm, sess = build_row_map(cfg, Calendars.load(list(cfg.calendars.required)), window_from_manifest(snap.manifest))
            out.append(_r("row_map", True, f"{len(rm.complete_case)} complete-case, {len(rm.eligible)} eligible rows, {len(rm.decisions)} monthly decisions"))
            out.extend(_r(n, ok, d) for n, ok, d in complete_case_coverage(cfg, snap, sess["XNYS"].intersection(sess["XKRX"])))
        except ValueError as e:  # too short a window, a ninth outer fold, ...: reported, never truncated
            out.append(_r("row_map", False, str(e)))

    # manifest integrity and fixture/research isolation
    manifest = json.loads((root / d.manifest).read_text(encoding="utf-8"))
    base = (root / d.manifest).parent
    bad = [n for n, m in manifest.get("files", {}).items()
           if "sha256:" + hashlib.sha256((base / n).read_bytes()).hexdigest() != m["raw_sha256"]]
    out.append(_r("manifest_integrity", not bad, "raw hashes match" if not bad else f"mismatch {bad}"))
    cross = [v for v in files.values() if ("snapshots" in v) == (d.mode == "fixture")]
    out.append(_r("path_isolation", not cross, f"{d.mode} profile reads only its own root" if not cross else f"cross-root {cross}"))

    raw = pd.read_parquet(root / d.prices_raw)
    tr = pd.read_parquet(root / d.prices_total_return)
    delay = parse_delay(cfg.availability.price_publication_delay)
    cals = Calendars.load(list(cfg.calendars.required))
    exch = {a.asset_id: a.exchange for a in (cfg.universe.assets or [])}
    ccy = {a.asset_id: a.currency for a in (cfg.universe.assets or [])}

    for name, df in (("raw", raw), ("total_return", tr)):
        dup = int(df.duplicated(["instrument_id", "event_timestamp_utc"]).sum())
        out.append(_r(f"{name}_unique_keys", dup == 0, f"{dup} duplicate (instrument, event) keys"))
        ev = pd.to_datetime(df.event_timestamp_utc, utc=True)
        mono = all(g.is_monotonic_increasing for _, g in ev.groupby(df.instrument_id))
        out.append(_r(f"{name}_monotonic_timestamps", bool(mono), "event timestamps increase per instrument"))
        nonpos = int((df.value <= 0).sum() + df.value.isna().sum())
        out.append(_r(f"{name}_positive_prices", nonpos == 0, f"{nonpos} non-positive or null values"))
        lag = pd.to_datetime(df.available_at_utc, utc=True) - ev
        out.append(_r(f"{name}_availability_not_before_event", bool((lag >= pd.Timedelta(0)).all()),
                      f"min lag {lag.min()}"))
        unknown = sorted(set(df.instrument_id) - set(exch))
        out.append(_r(f"{name}_known_instruments", not unknown, f"unknown {unknown}" if unknown else "all declared"))
    out.append(_r("raw_total_return_roles_separate", set(raw.field) == {"close"} and set(tr.field) == {"total_return_index"}
                  and not (len(raw) == len(tr) and (raw.value.values == tr.value.values).all()), "raw=close, total_return=total_return_index, values differ"))
    out.append(_r("currency_consistent", all((raw[raw.instrument_id == a].currency == c).all() for a, c in ccy.items()),
                  "row currency matches asset metadata"))

    # every observation is the official close of an official session of its own calendar
    mismatched = 0
    for a, g in raw.groupby("instrument_id"):
        for s, ts in zip(g.exchange_session, g.event_timestamp_utc, strict=True):
            mismatched += fmt(cals.close(exch[a], pd.Timestamp(s))) != ts
    out.append(_r("sessions_match_official_calendar", mismatched == 0, f"{mismatched} rows off the official close"))
    late = int(((pd.to_datetime(raw.available_at_utc, utc=True) - pd.to_datetime(raw.event_timestamp_utc, utc=True)) != delay).sum())
    out.append(_r("publication_delay_as_configured", late == 0, f"{late} rows differ from {cfg.availability.price_publication_delay}"))

    fx = pd.read_parquet(root / d.fx)
    out.append(_r("fx_quote_convention", set(fx.quote_convention) == {cfg.fx.quote_convention} and set(fx.pair_id) == {cfg.fx.pair_id}
                  and bool((fx.value > 0).all()), f"{cfg.fx.pair_id} {cfg.fx.quote_convention}, positive"))
    rates = pd.read_parquet(root / d.rates)
    out.append(_r("rate_basis", set(rates.quote_basis) == {cfg.cash.quote_basis} and set(rates.day_count) == {cfg.cash.day_count}
                  and bool((rates.annual_rate >= 0).all()), f"{cfg.cash.quote_basis} {cfg.cash.day_count}, non-negative"))

    ca = pd.read_parquet(root / d.corporate_actions)
    supported = {"split", "cash_dividend"}
    out.append(_r("corporate_actions_supported", set(ca.action_type) <= supported,
                  f"types {sorted(set(ca.action_type))}; unsupported types halt before the event (spec 4.3)"))
    out.append(_r("corporate_actions_declared_before_ex", bool((pd.to_datetime(ca.available_at_utc, utc=True)
                  < pd.to_datetime(ca.ex_timestamp_utc, utc=True)).all()), "available_at precedes ex timestamp"))

    from .reconciliation import reconcile

    out.extend(reconcile(cfg, root).results())  # spec 4.3 levels 1-3 and the halt scope

    try:
        cash, qty, as_of = load_initial_holdings(cfg, root)
        out.append(_r("initial_holdings", True, f"cash {cash}, quantities {qty}, as_of {as_of}"))
    except ValueError as e:
        out.append(_r("initial_holdings", False, str(e)))

    assets = cfg.universe.assets or []
    members = [a for a in assets if a.benchmark_eligible]
    for b in cfg.benchmarks.definitions:
        if b.type != "rules" or b.total_risky_weight is None:
            continue
        w = b.total_risky_weight
        ok = (len(members) >= cfg.benchmarks.required_benchmark_eligible_count
              and sum(a.max_weight for a in members) >= w and w / len(members) <= min(a.max_weight for a in members))
        out.append(_r(f"benchmark_feasible[{b.benchmark_id}]", ok, f"|M|={len(members)}, B={w}"))
    return out


FIXTURE_ASSET_IDS = frozenset({"KRX_A", "KRX_B", "NYSE_C"})


def validate_research_universe(cfg: Config, root: Path) -> list[Result]:
    """Research universe manifest (spec 4.6, 14/15 universe block): explicit records, 8-12 risky assets, required exchanges and
    currencies, no fixture asset ids, unique ids and a feasible benchmark set. Nothing is dropped or repaired silently."""
    from pydantic import ValidationError

    from .config import Asset, load_yaml_strict

    out: list[Result] = []
    u = cfg.universe
    path = root / (cfg.data.universe_manifest or "")
    if cfg.data.mode != "snapshot" or not cfg.data.universe_manifest:
        return [_r("universe_manifest", False, "a snapshot profile with data.universe_manifest is required")]
    if not path.exists():
        return [_r("universe_manifest", False, f"missing {cfg.data.universe_manifest}")]
    try:
        raw = load_yaml_strict(path.read_text(encoding="utf-8"))
        records = raw["assets"]
        assets = [Asset.model_validate(r) for r in records]
    except (ValueError, KeyError, TypeError, ValidationError, yaml.YAMLError) as e:  # malformed YAML is a reported failure, not a crash
        return [_r("universe_manifest_parse", False, (str(e).splitlines() or [type(e).__name__])[0])]
    ids = [a.asset_id for a in assets]
    out.append(_r("universe_records_explicit", u.require_explicit_asset_records is True and len(assets) == len(records),
                  f"{len(assets)} explicit asset records"))
    out.append(_r("universe_unique_ids", len(set(ids)) == len(ids), "asset ids are unique" if len(set(ids)) == len(ids) else "duplicate ids"))
    out.append(_r("universe_size_within_bounds", u.risky_asset_min_count <= len(assets) <= u.risky_asset_max_count,
                  f"{len(assets)} risky assets, required {u.risky_asset_min_count}-{u.risky_asset_max_count}"))
    missing_ex = sorted(set(u.required_exchanges or []) - {a.exchange for a in assets})
    missing_ccy = sorted(set(u.required_currencies or []) - {a.currency for a in assets})
    out.append(_r("universe_required_exchanges_and_currencies", not missing_ex and not missing_ccy,
                  f"missing exchanges {missing_ex}, currencies {missing_ccy}" if (missing_ex or missing_ccy) else "all required present"))
    leaked = sorted(set(ids) & FIXTURE_ASSET_IDS)
    out.append(_r("universe_rejects_fixture_asset_ids", not (u.reject_fixture_asset_ids and leaked),
                  f"fixture ids present: {leaked}" if leaked else "no fixture asset ids"))
    out.append(_r("universe_cash_is_not_a_risky_asset", u.cash_asset_id not in ids, f"{u.cash_asset_id} is the separate cash asset"))
    members = [a for a in assets if a.benchmark_eligible]
    for b in cfg.benchmarks.definitions:
        if b.type != "rules" or b.total_risky_weight is None:
            continue
        w = b.total_risky_weight
        ok = (len(members) >= cfg.benchmarks.required_benchmark_eligible_count and sum(a.max_weight for a in members) >= w
              and bool(members) and w / len(members) <= min(a.max_weight for a in members))
        out.append(_r(f"benchmark_feasible[{b.benchmark_id}]", ok, f"|M|={len(members)}, B={w}, "
                      f"required |M|>={cfg.benchmarks.required_benchmark_eligible_count}"))
    return out
