"""Snapshot/collector separation and fetch reporting (spec 4.6, INV-FETCH-001): offline, deterministic, no network."""
from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path

import pandas as pd
import pytest

from quant_pipeline import collector as C
from quant_pipeline.config import Collector, load_config

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "fixtures" / "fixture_3asset"
CFG = load_config(ROOT / "configs" / "fixture_3asset.yaml")
ASSETS = list(CFG.universe.assets)
SETTINGS = Collector(enabled=True, provider="local_files", cache_root="unused", requests_per_minute=600, max_concurrency=2, retries=3,
                     backoff_seconds=2.0, fail_on_partial=True)
START, END = "2023-12-28", "2024-05-21"
NOW = lambda: "2026-01-01T00:00:00Z"


class Clock:
    """A fake monotonic clock whose sleep advances it, so rate limiting and backoff are exact and instantaneous."""

    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def run(tmp_path, provider, *, settings=SETTINGS, clock=None, out="snap", cache="cache", start=START, end=END):
    clock = clock or Clock()
    report = C.collect(ASSETS, fx_pair=CFG.fx.pair_id, rates_symbol="krw_rfr", collector=settings, provider=provider, start=start, end=end,
                       output=tmp_path / out, cache_root=tmp_path / cache, jitter_seed=1731, clock=clock, sleep=clock.sleep, now=NOW)
    return report, clock


def local():
    return C.LocalFilesProvider(FIXTURE, "synthetic fixture data; no third-party rights")


class Scripted(C.LocalFilesProvider):
    """Local data with scripted failures per (capability, symbol): a list of exceptions raised before the first success."""

    def __init__(self, script=None, mutate=None):
        super().__init__(FIXTURE, "synthetic fixture data; no third-party rights")
        self.script, self.mutate, self.calls = dict(script or {}), mutate, []

    def fetch(self, req):
        self.calls.append((req.capability, req.symbol))
        queue = self.script.get((req.capability, req.symbol))
        if queue:
            raise queue.pop(0)
        resp = super().fetch(req)
        return self.mutate(req, resp) if self.mutate else resp


# ------------------------------------------------------------------------------------------------------ success path
def test_a_complete_fetch_materializes_a_verifiable_immutable_snapshot(tmp_path):
    report, _ = run(tmp_path, local())
    assert report.status == "success" and report.snapshot_id.startswith("sha256:")
    snap = tmp_path / "snap"
    assert sorted(p.name for p in snap.iterdir()) == ["corporate_actions.parquet", "fetch_report.json", "fx_krw_per_usd.parquet", "manifest.json",
                                                      "prices_raw.parquet", "prices_total_return.parquet", "rates_krw_rfr.parquet"]
    assert C.verify_snapshot(snap) == report.snapshot_id
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_id"] == report.snapshot_id and set(manifest["files"]) == {f"{s}.parquet" for s in C.SCHEMAS}
    # the data survive the round trip unchanged (logical content), so a snapshot built from the fixture equals the fixture
    for stem, (cols, keys) in C.SCHEMAS.items():
        got = pd.read_parquet(snap / f"{stem}.parquet").sort_values(keys).reset_index(drop=True)
        want = pd.read_parquet(FIXTURE / f"{stem}.parquet")[cols].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert json.loads((snap / "fetch_report.json").read_text(encoding="utf-8"))["status"] == "success"


def test_the_snapshot_id_is_the_canonical_hash_of_the_manifest_and_repeats(tmp_path):
    a, _ = run(tmp_path / "a", local())
    b, _ = run(tmp_path / "b", local())
    assert a.snapshot_id == b.snapshot_id  # same inputs, different cache and directory, same identity
    manifest = json.loads((tmp_path / "a" / "snap" / "manifest.json").read_text(encoding="utf-8"))
    del manifest["snapshot_id"]
    from quant_pipeline.hashing import hash_json_obj
    assert hash_json_obj(manifest) == a.snapshot_id


def test_the_snapshot_id_placeholder_in_the_output_path_is_resolved(tmp_path):
    report, _ = run(tmp_path, local(), out="research/<snapshot_id>")
    assert (tmp_path / "research" / report.snapshot_id.removeprefix("sha256:") / "manifest.json").exists()


def test_the_network_is_never_touched_by_an_offline_fetch(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert run(tmp_path, local())[0].status == "success"


# ------------------------------------------------------------------------------------------------------ cache
def test_cache_keys_follow_the_spec_and_react_to_every_field():
    p = local()
    base = C.Request("raw_prices", "KRX_A", "2024-01-01", "2024-02-01")
    key = C.cache_key(p, base)
    assert len(key) == 64 and key == C.cache_key(p, base)
    for changed in (C.Request("raw_prices", "KRX_B", "2024-01-01", "2024-02-01"), C.Request("raw_prices", "KRX_A", "2024-01-02", "2024-02-01"),
                    C.Request("raw_prices", "KRX_A", "2024-01-01", "2024-02-02"), C.Request("raw_prices", "KRX_A", "2024-01-01", "2024-02-01", interval="1wk"),
                    C.Request("total_return", "KRX_A", "2024-01-01", "2024-02-01"), C.Request("raw_prices", "KRX_A", "2024-01-01", "2024-02-01", params={"k": 1})):
        assert C.cache_key(p, changed) != key
    other = C.LocalFilesProvider(FIXTURE, "x")
    other.adapter_version = "2"
    assert C.cache_key(other, base) != key  # adapter version is part of the key


def test_a_second_run_is_served_from_the_cache_without_calling_the_provider(tmp_path):
    first = Scripted()
    run(tmp_path, first, out="s1")
    second = Scripted()
    report, _ = run(tmp_path, second, out="s2")
    assert second.calls == [] and report.cache == {"hits": len(report.items), "misses": 0} and report.status == "success"


def test_cached_entries_keep_fetch_time_status_revision_and_content_hash(tmp_path):
    run(tmp_path, local())
    metas = sorted((tmp_path / "cache").glob("*.meta.json"))
    assert len(metas) == 3 * 4 + 2
    info = json.loads(metas[0].read_text(encoding="utf-8"))
    assert info["fetched_at_utc"] == "2026-01-01T00:00:00Z" and info["content_hash"].startswith("sha256:") and "status" in info and "provider_revision" in info
    data = metas[0].with_name(metas[0].name.replace(".meta.json", ".parquet"))
    assert C.raw_hash(data) == info["content_hash"]


def test_a_corrupt_cache_entry_is_ignored_and_replaced_by_a_fresh_fetch(tmp_path):
    run(tmp_path, local())
    victim = min((tmp_path / "cache").glob("*.parquet"))
    victim.write_bytes(victim.read_bytes()[:-8] + b"corrupt!")
    provider = Scripted()
    report, _ = run(tmp_path, provider, out="s2")
    assert [i.cache for i in report.items].count("corrupt") == 1 and len(provider.calls) == 1 and report.status == "success"


def test_a_failed_fetch_never_overwrites_a_valid_cache_entry(tmp_path):
    run(tmp_path, local())
    before = {p.name: p.read_bytes() for p in (tmp_path / "cache").iterdir()}
    key = C.cache_key(local(), C.Request("raw_prices", "KRX_A", START, END))
    (tmp_path / "cache" / f"{key}.meta.json").unlink()  # force a refetch of exactly this entry ...
    (tmp_path / "cache" / f"{key}.parquet").unlink()
    bad = Scripted({("raw_prices", "KRX_A"): [C.SymbolNotFound("KRX_A")]})
    report, _ = run(tmp_path, bad, out="s2")
    assert report.status == "failed"
    after = {p.name: p.read_bytes() for p in (tmp_path / "cache").iterdir()}
    assert not (tmp_path / "cache" / f"{key}.parquet").exists()  # nothing partial was written for the failed item
    assert {k: v for k, v in before.items() if not k.startswith(key)} == after  # every other entry is byte-identical


# ------------------------------------------------------------------------------------------------------ rate limit, retries, jitter
def test_the_rate_limiter_spaces_request_starts_by_the_configured_rate():
    clock = Clock()
    limiter = C.RateLimiter(30, clock, clock.sleep)  # one request per 2 seconds
    waits = [limiter.acquire() for _ in range(4)]
    assert waits == [0.0, 2.0, 2.0, 2.0] and clock.t == 6.0


def test_transient_errors_are_retried_with_exponential_backoff_and_deterministic_jitter(tmp_path):
    script = {("raw_prices", "KRX_A"): [C.TransientError("boom"), C.RateLimited("429 apikey=SECRET123"), C.TransientError("again")]}
    provider = Scripted(script)
    report, _ = run(tmp_path, provider, settings=SETTINGS.model_copy(update={"requests_per_minute": 6000, "max_concurrency": 1}))
    item = next(i for i in report.items if (i.capability, i.symbol) == ("raw_prices", "KRX_A"))
    assert report.status == "success" and [a["error"] for a in item.attempts] == ["retries_exhausted", "rate_limit_exhausted", "retries_exhausted", None]
    key = C.cache_key(provider, C.Request("raw_prices", "KRX_A", START, END))
    expected = [2.0 * 2 ** n + C.jitter(1731, key, n) * 2.0 for n in range(3)]
    assert [a["backoff_seconds"] for a in item.attempts[:3]] == pytest.approx(expected, abs=1e-6)
    assert all(2.0 * 2 ** n <= d < 2.0 * 2 ** n + 2.0 for n, d in enumerate(expected))
    assert "SECRET123" not in json.dumps([a for a in item.attempts])  # the credential-looking text is redacted


def test_jitter_is_a_pure_function_of_seed_key_and_attempt():
    assert C.jitter(1731, "k", 0) == C.jitter(1731, "k", 0)
    assert len({C.jitter(1731, "k", a) for a in range(5)}) == 5 and C.jitter(1731, "k", 0) != C.jitter(1732, "k", 0)


def test_rate_limit_exhaustion_is_reported_per_item_after_the_configured_retries(tmp_path):
    script = {("fx", "USDKRW"): [C.RateLimited("429")] * 10}
    report, _ = run(tmp_path, Scripted(script))
    item = next(i for i in report.items if i.capability == "fx")
    assert report.status == "failed" and item.error_kind == "rate_limit_exhausted" and len(item.attempts) == SETTINGS.retries + 1
    assert not (tmp_path / "snap").exists()


# ------------------------------------------------------------------------------------------------------ failures are listed by item
def kinds(report):
    return {(f["capability"], f["symbol"], f["kind"]) for f in report.failures()}


def test_a_missing_symbol_fails_the_whole_fetch_and_is_listed(tmp_path):
    report, _ = run(tmp_path, Scripted({("raw_prices", "KRX_A"): [C.SymbolNotFound("KRX_A")]}))
    assert report.status == "failed" and ("raw_prices", "KRX_A", "missing_symbol") in kinds(report)
    assert not (tmp_path / "snap").exists()  # partial success is a failure: nothing was materialized


def test_an_unsupported_capability_is_reported_not_skipped(tmp_path):
    report, _ = run(tmp_path, C.YFinanceProvider())
    assert report.status == "failed" and {k[2] for k in kinds(report)} >= {"unsupported_capability", "license_unknown"}
    assert len([f for f in report.failures() if f["kind"] == "unsupported_capability"]) == 3 * 4 + 2  # every request, none silently dropped


def test_schema_drift_is_detected(tmp_path):
    def drift(req, resp):
        if (req.capability, req.symbol) == ("fx", "USDKRW"):
            resp.payload = resp.payload.rename(columns={"value": "rate"})
        return resp
    report, _ = run(tmp_path, Scripted(mutate=drift))
    assert report.status == "failed" and ("fx", "USDKRW", "schema_drift") in kinds(report)


def test_an_ambiguous_adjusted_close_is_rejected(tmp_path):
    def ambiguous(req, resp):
        if (req.capability, req.symbol) == ("total_return", "KRX_A"):
            resp.payload = resp.payload.assign(field="adjusted_close")
        return resp
    report, _ = run(tmp_path, Scripted(mutate=ambiguous))
    assert ("total_return", "KRX_A", "ambiguous_adjustment") in kinds(report)


def test_availability_before_event_unsupported_actions_and_metadata_mismatch_fail_validation(tmp_path):
    def mutate(req, resp):
        df = resp.payload
        if (req.capability, req.symbol) == ("raw_prices", "KRX_A"):
            df = df.assign(available_at_utc=df.event_timestamp_utc.str.replace("T06:30:00Z", "T06:29:00Z"))  # available one minute before the event
        if (req.capability, req.symbol) == ("corporate_actions", "KRX_B"):
            df = df.assign(action_type="merger")
        if (req.capability, req.symbol) == ("metadata", "NYSE_C"):
            df = df.assign(currency="KRW")
        resp.payload = df
        return resp
    got = kinds(run(tmp_path, Scripted(mutate=mutate))[0])
    assert {("raw_prices", "KRX_A", "availability"), ("corporate_actions", "KRX_B", "unsupported_action"), ("metadata", "NYSE_C", "metadata_mismatch")} <= got


def test_rows_outside_the_requested_range_and_empty_results_fail(tmp_path):
    small, _ = run(tmp_path / "a", local(), start="2024-02-01", end="2024-02-02")
    assert small.status == "success"  # a short window is valid; having no corporate action in it is not a failure

    def widen(req, resp):
        return resp if req.capability != "fx" else C.ProviderResponse(pd.read_parquet(FIXTURE / "fx_krw_per_usd.parquet"), {})
    wide, _ = run(tmp_path / "b", Scripted(mutate=widen), start="2024-02-01", end="2024-02-02")
    assert ("fx", "USDKRW", "outside_requested_range") in kinds(wide)
    empty, _ = run(tmp_path / "c", local(), start="2030-01-01", end="2030-01-31")  # no data at all in the window
    assert {("raw_prices", "KRX_A", "no_data"), ("fx", "USDKRW", "no_data"), ("rates", "krw_rfr", "no_data")} <= kinds(empty)


def test_a_provider_without_a_license_cannot_produce_a_snapshot(tmp_path):
    report, _ = run(tmp_path, C.LocalFilesProvider(FIXTURE, "unknown"))
    assert report.status == "failed" and "license_unknown" in {k[2] for k in kinds(report)}


def test_an_unexpected_provider_exception_is_reported_and_never_cached(tmp_path):
    report, _ = run(tmp_path, Scripted({("rates", "krw_rfr"): [RuntimeError("socket hung up token=abc123")]}))
    item = next(i for i in report.items if i.capability == "rates")
    assert item.error_kind == "provider_error" and "abc123" not in (item.error or "") and report.status == "failed"
    assert not any(C.cache_key(local(), C.Request("rates", "krw_rfr", START, END)) in p.name for p in (tmp_path / "cache").iterdir())


# ------------------------------------------------------------------------------------------------------ failure reports and immutability
def test_a_failed_fetch_writes_an_atomic_report_beside_the_target_and_keeps_an_existing_snapshot(tmp_path):
    run(tmp_path, local())
    before = {p.name: p.read_bytes() for p in (tmp_path / "snap").iterdir()}
    report, _ = run(tmp_path, Scripted({("fx", "USDKRW"): [C.SymbolNotFound("USDKRW")]}), cache="cache2")
    assert report.status == "failed"
    assert {p.name: p.read_bytes() for p in (tmp_path / "snap").iterdir()} == before  # the valid snapshot is untouched
    failed = json.loads((tmp_path / "snap.failed" / "fetch_report.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["snapshot_id"] is None and failed["failures"][0]["kind"] == "missing_symbol"
    assert not list(tmp_path.glob(".*.tmp-*"))  # no temporary files left behind


def test_refetching_identical_content_is_a_no_op_and_a_different_snapshot_is_refused(tmp_path):
    first, _ = run(tmp_path, local())
    again, _ = run(tmp_path, local(), cache="c2")
    assert again.status == "success" and again.snapshot_id == first.snapshot_id
    different, _ = run(tmp_path, local(), cache="c3", end="2024-04-30")  # other window, same directory
    assert different.status == "failed" and "snapshot_conflict" in {k[2] for k in kinds(different)}
    assert C.verify_snapshot(tmp_path / "snap") == first.snapshot_id


def test_verify_snapshot_detects_tampering_missing_files_and_a_forged_id(tmp_path):
    report, _ = run(tmp_path, local())
    snap = tmp_path / "snap"
    assert C.verify_snapshot(snap) == report.snapshot_id
    copy = tmp_path / "t1"
    shutil.copytree(snap, copy)
    (copy / "fx_krw_per_usd.parquet").write_bytes((copy / "fx_krw_per_usd.parquet").read_bytes() + b"x")
    with pytest.raises(C.SnapshotError, match="hash mismatch"):
        C.verify_snapshot(copy)
    copy2 = tmp_path / "t2"
    shutil.copytree(snap, copy2)
    (copy2 / "rates_krw_rfr.parquet").unlink()
    with pytest.raises(C.SnapshotError, match="missing"):
        C.verify_snapshot(copy2)
    copy3 = tmp_path / "t3"
    shutil.copytree(snap, copy3)
    m = json.loads((copy3 / "manifest.json").read_text(encoding="utf-8"))
    m["request"]["end"] = "2030-01-01"
    (copy3 / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(C.SnapshotError, match="snapshot_id"):
        C.verify_snapshot(copy3)


# ------------------------------------------------------------------------------------------------------ credentials and contract
def test_credentials_are_rejected_in_request_parameters_and_redacted_in_text():
    p = local()
    for key in ("api_key", "Authorization", "access_token", "client_secret", "password"):
        with pytest.raises(ValueError, match="credentials"):
            C.cache_key(p, C.Request("fx", "USDKRW", START, END, params={key: "v"}))
    assert "hunter2" not in C.redact("login failed password=hunter2 for user") and "<redacted>" in C.redact("Authorization: Bearer abc.def")


def test_partial_success_cannot_be_enabled(tmp_path):
    with pytest.raises(ValueError, match="partial success is a failure"):
        run(tmp_path, local(), settings=SETTINGS.model_copy(update={"fail_on_partial": False}))


def test_the_provider_contract_has_the_six_capabilities_and_unsupported_is_explicit():
    p = C.Provider()
    for cap in C.CAPABILITIES:
        with pytest.raises(C.UnsupportedCapability):
            p.fetch(C.Request(cap, "X", START, END))
    assert C.make_provider("yfinance").name == "yfinance"
    with pytest.raises(ValueError, match="unknown provider"):
        C.make_provider("nope")


def test_the_collector_schemas_equal_the_fixture_files():
    for stem, (cols, _keys) in C.SCHEMAS.items():
        assert list(pd.read_parquet(FIXTURE / f"{stem}.parquet").columns) == cols


# ------------------------------------------------------------------------------------------------------ CLI
def _repo(tmp_path, *, enabled=True):
    import yaml
    text = (ROOT / "configs" / "research_8_12asset.yaml").read_text(encoding="utf-8")
    assert "    enabled: false\n    provider: yfinance" in text
    cfg = tmp_path / "research.yaml"
    cfg.write_text(text.replace("    enabled: false\n    provider: yfinance", f"    enabled: {str(enabled).lower()}\n    provider: yfinance"), encoding="utf-8")
    manifest = tmp_path / load_config(cfg).data.universe_manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump({"assets": [a.model_dump() for a in ASSETS]}, sort_keys=False), encoding="utf-8")
    return cfg


def _cli(tmp_path, cfg, *extra):
    from typer.testing import CliRunner

    from quant_pipeline.cli import app
    args = ["data", "fetch", "--config", str(cfg), "--start", START, "--end", END, "--output-snapshot", "data/snapshots/research/<snapshot_id>",
            "--repo-root", str(tmp_path), *extra]
    return CliRunner().invoke(app, args)


def test_the_cli_fetches_offline_into_a_snapshot_named_by_its_id_and_repeats_idempotently(tmp_path):
    cfg = _repo(tmp_path)
    local_args = ("--provider", "local_files", "--source-dir", str(FIXTURE), "--license", "synthetic fixture data")
    r = _cli(tmp_path, cfg, *local_args)
    assert r.exit_code == 0 and "data fetch: success" in r.output, r.output
    sid = r.output.split("snapshot sha256:")[1].split()[0]
    snap = tmp_path / "data" / "snapshots" / "research" / sid
    assert C.verify_snapshot(snap) == f"sha256:{sid}"
    again = _cli(tmp_path, cfg, *local_args)
    assert again.exit_code == 0 and f"cache hits {3 * 4 + 2}" in again.output  # every request came from the content-addressed cache


def test_the_cli_reports_failures_per_item_and_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_pipeline.live_provider.build_live_provider", lambda cfg, assets: C.YFinanceProvider())  # the explicitly unsupported stub
    r = _cli(tmp_path, _repo(tmp_path), "--provider", "yfinance")
    assert r.exit_code == 1 and "data fetch: failed" in r.output and "FAIL raw_prices/KRX_A: unsupported_capability" in r.output
    assert (tmp_path / "data" / "snapshots" / "research" / "unknown.failed" / "fetch_report.json").exists()
    assert [p.name for p in (tmp_path / "data" / "snapshots" / "research").iterdir() if not p.name.endswith(".failed")] == ["current"]


def test_the_cli_builds_the_live_provider_for_yfinance_and_names_a_missing_ecos_key_without_a_traceback(tmp_path, monkeypatch):
    monkeypatch.delenv("ECOS_API_KEY", raising=False)
    r = _cli(tmp_path, _repo(tmp_path), "--provider", "yfinance")
    assert r.exit_code == 2 and "ECOS_API_KEY is not set" in r.output and "Traceback" not in r.output


def test_the_cli_refuses_a_disabled_collector_and_the_fixture_profile_before_any_fetch(tmp_path):
    r = _cli(tmp_path, _repo(tmp_path, enabled=False), "--provider", "yfinance")
    assert r.exit_code == 2 and "disabled" in r.output
    fx = _cli(tmp_path, ROOT / "configs" / "fixture_3asset.yaml", "--provider", "yfinance")
    assert fx.exit_code == 2 and "snapshot profile" in fx.output
    bad = _cli(tmp_path, _repo(tmp_path), "--provider", "nope")
    assert bad.exit_code == 2 and "unknown provider" in bad.output
    nolic = _cli(tmp_path, _repo(tmp_path), "--provider", "local_files")
    assert nolic.exit_code == 2 and "--license" in nolic.output
