"""Command-line interface (spec 13). Implemented so far: ``config validate``, static ``doctor``, ``data validate``.

Commands that need later phases (plan, run, evaluate, holdout, report, resume, rebalance, data fetch)
exit with code 2 and a clear message; they never pretend to work.
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from . import SPEC_VERSION
from .config import Config, load_config, load_yaml_strict
from .hashing import hash_json_obj, hash_text

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Heterogeneous-agent quant research pipeline")
data_app = typer.Typer(help="data commands")
app.add_typer(data_app, name="data")
config_app = typer.Typer(help="configuration commands")
app.add_typer(config_app, name="config")
holdout_app = typer.Typer(help="sealed final holdout")
app.add_typer(holdout_app, name="holdout")

REQUIRED_IMPORTS = ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "hmmlearn", "exchange_calendars", "cvxpy",
                    "osqp", "pydantic", "yaml", "rfc8785", "jinja2", "typer"]
OFFLINE_ONLY_FORBIDDEN_IN_FIXTURE = ["yfinance"]
Check = dict[str, object]
CONFIG_OPT = Annotated[Path, typer.Option("--config", help="path to a profile YAML")]


def _check(name: str, ok: bool, detail: str, blocking: bool = True) -> Check:
    return {"check": name, "status": "pass" if ok else "fail", "blocking": blocking, "detail": detail}


def static_doctor(config_path: Path, repo_root: Path) -> dict[str, object]:
    checks: list[Check] = []
    try:
        cfg = load_config(config_path)
        checks.append(_check("config_schema", True, f"{cfg.profile.name} ({cfg.profile.profile_type})"))
    except Exception as e:  # noqa: BLE001
        checks.append(_check("config_schema", False, str(e).splitlines()[0]))
        return _finish(checks, config_path, None)

    py = sys.version_info
    checks.append(_check("python_version", py[:2] == (3, 12),
                         f"{platform.python_version()} (spec target 3.12; deviation D1 in PROGRESS.md)", blocking=False))
    missing = []
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod}: {e}")
    checks.append(_check("imports", not missing, "; ".join(missing) or f"{len(REQUIRED_IMPORTS)} modules import"))

    try:
        import exchange_calendars as xc
        for cal in cfg.calendars.required:
            xc.get_calendar(cal)
        checks.append(_check("calendars", True, f"{cfg.calendars.required} resolve (exchange_calendars {xc.__version__})"))
    except Exception as e:  # noqa: BLE001
        checks.append(_check("calendars", False, str(e)))

    lock = repo_root / "uv.lock"
    checks.append(_check("accepted_lock", lock.exists(),
                         "uv.lock present" if lock.exists() else "uv.lock absent (created only after target-VM verification)",
                         blocking=False))
    checks.append(_check("numeric_threads", cfg.reproducibility.numeric_threads == 1 or cfg.profile.profile_type == "research",
                         f"numeric_threads={cfg.reproducibility.numeric_threads}"))

    d = cfg.data
    paths = [d.manifest, d.prices_raw, d.prices_total_return, d.fx, d.rates, d.corporate_actions, d.initial_holdings]
    if d.mode == "fixture":
        paths += [d.generation_lock or "", cfg.regime.fixed_table.path if cfg.regime.fixed_table else ""]
    absent = [p for p in paths if p and not (repo_root / p).exists()]
    checks.append(_check("data_files", not absent, "all present" if not absent else f"missing: {absent}",
                         blocking=d.mode == "fixture"))

    if d.mode == "fixture" and not absent:
        manifest = json.loads((repo_root / d.manifest).read_text(encoding="utf-8"))
        bad = []
        for name, meta in manifest["files"].items():
            import hashlib
            h = "sha256:" + hashlib.sha256((repo_root / Path(d.manifest).parent / name).read_bytes()).hexdigest()
            if h != meta["raw_sha256"]:
                bad.append(name)
        checks.append(_check("fixture_manifest_hashes", not bad, "raw hashes match manifest" if not bad else f"mismatch: {bad}"))
        fixture_roots = {str(Path(p).parent) for p in paths if p}
        checks.append(_check("fixture_isolation", all("snapshots" not in p for p in paths), f"roots: {sorted(fixture_roots)}"))
        checks.append(_check("collector_not_loaded", "yfinance" not in sys.modules,
                             "no optional collector module imported by the fixture path"))

    out_root = repo_root / cfg.paths.output_root
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        probe = out_root / ".doctor_write_probe"
        probe.write_text("x", encoding="utf-8")
        probe.replace(out_root / ".doctor_write_probe.renamed")
        (out_root / ".doctor_write_probe.renamed").unlink()
        checks.append(_check("filesystem_write_rename", True, str(out_root)))
    except OSError as e:
        checks.append(_check("filesystem_write_rename", False, str(e)))

    spec_path = repo_root / "IMPLEMENTATION_SPEC.md"
    if spec_path.exists():
        checks.append(_check("spec_hash", True, hash_text(spec_path.read_bytes())))
    else:
        checks.append(_check("spec_hash", False, "IMPLEMENTATION_SPEC.md missing"))
    checks.append(_check("jcs_probe", hash_json_obj({"b": 1, "a": [1, 2]}) == hash_json_obj({"a": [1, 2], "b": 1}),
                         "key-order independent JCS hash"))
    checks.append(_check("clock_parsing", datetime.now(UTC).tzinfo is not None, "timezone-aware UTC clock"))
    from .capability import probe as capability_probe

    cap = capability_probe(repo_root)
    sha = cap["git"]["head_sha"]
    checks.append(_check("git_truth", True, f"HEAD {sha['value']}" if sha["value"] else f"commit SHA is explicitly null: {sha['reason']}",
                         blocking=False))
    checks.append(_check("github_capability", True, f"gh installed: {cap['github_cli']['gh_installed']}; in CI: {cap['ci']['in_ci']}; "
                         f"SHA audit: {cap['sha_audit']['status']}", blocking=False))
    return _finish(checks, config_path, cfg)


def _finish(checks: list[Check], config_path: Path, cfg: Config | None) -> dict[str, object]:
    blocking_fail = [c for c in checks if c["status"] == "fail" and c["blocking"]]
    return {
        "doctor": "static",
        "config": str(config_path),
        "spec_version": SPEC_VERSION,
        "approves_compute_budget": False,  # static success is necessary, not a budget approval (spec 18)
        "status": "fail" if blocking_fail else "pass",
        "checks": checks,
    }


@config_app.command("validate")
def config_validate(config: CONFIG_OPT) -> None:
    """Strict parse, canonicalize, hash and print the resolved configuration."""
    try:
        raw = load_yaml_strict(config.read_text(encoding="utf-8"))
        cfg = Config.model_validate(raw)
    except Exception as e:
        typer.echo(f"INVALID: {e}", err=True)
        raise typer.Exit(1) from e
    resolved = cfg.model_dump(mode="json")
    typer.echo(json.dumps({"resolved_config_hash": hash_json_obj(resolved), "profile": cfg.profile.name}, indent=2))


@app.command()
def doctor(
    config: CONFIG_OPT,
    after_plan: Annotated[bool, typer.Option("--after-plan", help="requires a compute plan (later phase)")] = False,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path(),
) -> None:
    """Static pre-plan validation. It does not approve the compute budget."""
    if after_plan:
        if not run_id:
            typer.echo("doctor --after-plan requires --run-id", err=True)
            raise typer.Exit(2)
        raise typer.Exit(_doctor_after_plan(config, run_id, repo_root.resolve()))
    report = static_doctor(config, repo_root.resolve())
    root = repo_root.resolve()
    target = root / "outputs" / "doctor.static.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for c in report["checks"]:  # type: ignore[union-attr]
        flag = "ok  " if c["status"] == "pass" else ("FAIL" if c["blocking"] else "warn")
        typer.echo(f"{flag} {c['check']}: {c['detail']}")
    typer.echo(f"doctor.static: {report['status']} (does not approve compute budget) -> {target}")
    raise typer.Exit(0 if report["status"] == "pass" else 1)


@app.command()
def probe(repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path()) -> None:
    """Capability probe and Git/GitHub/CI provenance audit: observed facts or explicit nulls, never fabricated."""
    from .capability import probe as capability_probe

    root = repo_root.resolve()
    result = capability_probe(root)
    target = root / "outputs" / "capability_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    typer.echo(f"sha audit: {result['sha_audit']['status']}; ci: {result['ci']['in_ci']}; wrote {target}")


def _not_implemented(name: str) -> None:
    typer.echo(f"`{name}` is not implemented yet (later phase).", err=True)
    raise typer.Exit(2)


def _after_plan_inputs(config: Path, run_id: str, root: Path):
    from .plan import write_plan  # noqa: F401  (imported for symmetry with `plan`)

    cfg = load_config(config)
    plan_path = root / cfg.paths.output_root / run_id / "compute_plan.json"
    return cfg, plan_path


def _doctor_after_plan(config: Path, run_id: str, root: Path) -> int:
    from .plan import doctor_after_plan

    cfg, plan_path = _after_plan_inputs(config, run_id, root)
    if not plan_path.exists():
        typer.echo(f"no compute_plan.json for run {run_id!r}; run `plan` first", err=True)
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = doctor_after_plan(cfg, plan, plan_path, root / cfg.paths.output_root, run_id)
    for c in report["checks"]:
        flag = "ok  " if c["status"] == "pass" else ("FAIL" if c["blocking"] else "warn")
        typer.echo(f"{flag} {c['check']}: {c['detail']}")
    typer.echo(f"doctor.after_plan: {report['status']}" + (f" - failed: {', '.join(report['failed'])}" if report["failed"] else ""))
    return 0 if report["status"] == "pass" else 1


@app.command()
def plan(config: CONFIG_OPT, run_id: Annotated[str | None, typer.Option("--run-id")] = None,
         repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path()) -> None:
    """Materialize the graph-derived compute plan without fitting, solving or opening the holdout."""
    from .calendars import Calendars, build_row_map
    from .fitstate import lineage_hashes
    from .plan import build_compute_plan, write_plan
    from .splits import build_split_manifest
    from .trials import enumerate_trials

    root = repo_root.resolve()
    cfg = load_config(config)
    window = None
    snap = None
    if cfg.data.mode != "fixture":
        from .calendars import window_from_manifest
        from .snapshot import SnapshotResolutionError, load_research_context

        try:
            cfg, snap = load_research_context(cfg, root)
        except SnapshotResolutionError as e:
            typer.echo(f"plan refused: {e}", err=True)
            raise typer.Exit(2) from e
        window = window_from_manifest(snap.manifest)
    cals = Calendars.load(list(cfg.calendars.required))
    try:
        unobserved = frozenset()
        if snap is not None:
            from .snapshot import research_unobserved

            unobserved = research_unobserved(cfg, snap, cals)
        rm, _ = build_row_map(cfg, cals, window, unobserved=unobserved)
    except ValueError as e:
        typer.echo(f"plan refused: {e}", err=True)
        raise typer.Exit(2) from e
    manifest = build_split_manifest(cfg, cals, rm)
    plan_doc = build_compute_plan(cfg, manifest, enumerate_trials(cfg), lineage_hashes(cfg, root))
    path = write_plan(root / cfg.paths.output_root, run_id or "plan", plan_doc)
    typer.echo(f"wrote {path} (plan_hash {plan_doc['plan_hash'][:19]})")
    for unit, c in plan_doc["counts"].items():
        typer.echo(f"  {unit}: planned {c['deduplicated']} (no-sharing bound {c['no_sharing_upper_bound']}), cap {c['cap']}")


@app.command()
def run(config: CONFIG_OPT, run_id: Annotated[str, typer.Option("--run-id")] = "run",
        repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path()) -> None:
    """Execute the fixture pipeline, but only after a passing after-plan gate for exactly this plan (spec 11.8, 18)."""
    root = repo_root.resolve()
    cfg, plan_path = _after_plan_inputs(config, run_id, root)
    report_path = plan_path.parent / "doctor.after_plan.json"
    if not plan_path.exists() or not report_path.exists():
        typer.echo("refusing to run: plan and doctor --after-plan must both exist for this run id", err=True)
        raise typer.Exit(2)
    plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("plan_hash") != plan_doc["plan_hash"]:
        typer.echo("refusing to run: the after-plan report belongs to a different plan (a changed plan invalidates it)", err=True)
        raise typer.Exit(2)
    if report.get("status") != "pass":
        typer.echo(f"refusing to run: the after-plan gate failed ({', '.join(report.get('failed', []))}). "
                   "Caps are independent ceilings; changing them needs a recorded canonical revision.", err=True)
        raise typer.Exit(1)
    from .pipeline import FixturePipeline

    proj = FixturePipeline(root, root / cfg.paths.output_root, run_id, config=str(config)).run()
    (plan_path.parent / "projection.json").write_text(json.dumps(proj, sort_keys=True), encoding="utf-8", newline="\n")
    typer.echo(f"run complete: winners {proj['winners']}")


@holdout_app.command("open")
def holdout_open(config: CONFIG_OPT, run_id: Annotated[str, typer.Option("--run-id")],
                 unlock_final_holdout: Annotated[bool, typer.Option("--unlock-final-holdout")] = False,
                 allow_synthetic: Annotated[bool, typer.Option("--allow-synthetic", help="engineering tests only: accept a synthetic research snapshot")] = False,
                 repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path()) -> None:
    """Open the sealed holdout once. Needs the flag, the acknowledgement variable and a complete pre-holdout report."""
    from .holdout import HoldoutRefused, open_holdout
    from .pipeline import FixturePipeline

    root = repo_root.resolve()
    cfg = load_config(config)
    if cfg.regime.mode == "hmm_filtered":  # the research lane: snapshot, HMM-filtered regimes
        from .research_pipeline import ResearchPipeline, SyntheticSnapshotRefused

        try:
            pipe = ResearchPipeline(root, root / cfg.paths.output_root, run_id, config=str(config), allow_synthetic=allow_synthetic)
        except SyntheticSnapshotRefused as e:
            typer.echo(f"holdout NOT opened: {e}", err=True)
            raise typer.Exit(1) from e
    else:
        pipe = FixturePipeline(root, root / cfg.paths.output_root, run_id, config=str(config))
    try:
        result = open_holdout(pipe, unlock=unlock_final_holdout)
    except HoldoutRefused as e:
        typer.echo(f"holdout NOT opened: {e.reason}", err=True)
        raise typer.Exit(1) from e
    a = result["audit"]
    typer.echo(f"holdout opened once; audit {a['audit_hash'][:19]}; minima met: {a['minima']['met']}")


@data_app.command("validate")
def data_validate(config: CONFIG_OPT, repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path()) -> None:
    """Validate schemas, manifests, calendars, availability, isolation and benchmark feasibility (spec 4.5)."""
    from .data_quality import validate_data

    cfg = load_config(config)
    results = validate_data(cfg, repo_root.resolve())
    failed = [r for r in results if r["status"] != "pass"]
    for r in results:
        typer.echo(f"{'ok  ' if r['status'] == 'pass' else 'FAIL'} {r['check']}: {r['detail']}")
    typer.echo(f"data validate: {'fail' if failed else 'pass'} ({len(results)} checks)")
    raise typer.Exit(1 if failed else 0)


@data_app.command("fetch")
def data_fetch(
    config: CONFIG_OPT,
    provider: Annotated[str, typer.Option("--provider", help="yfinance (Yahoo prices and FX plus the ECOS overnight rate; needs ECOS_API_KEY and the research-online extra) or local_files")],
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")],
    output_snapshot: Annotated[Path, typer.Option("--output-snapshot", help="snapshot directory; <snapshot_id> is replaced by the manifest id")],
    source_dir: Annotated[Path | None, typer.Option("--source-dir", help="local_files only: directory of canonical parquet files")] = None,
    license_text: Annotated[str | None, typer.Option("--license", help="local_files only: license of the source data")] = None,
    repo_root: Annotated[Path, typer.Option("--repo-root", hidden=True)] = Path(),
) -> None:
    """Optional collector (spec 4.6): fetch, cache, validate and materialize an immutable research snapshot. Partial success is a failure."""
    from .collector import ProviderError, SnapshotError, assets_from_manifest, collect, make_provider

    cfg = load_config(config)
    root = repo_root.resolve()
    col = cfg.data.collector
    if cfg.data.mode != "snapshot" or col is None or not cfg.data.universe_manifest:
        typer.echo("data fetch needs a snapshot profile with data.collector and data.universe_manifest", err=True)
        raise typer.Exit(2)
    if not col.enabled:
        typer.echo("the collector is disabled in this profile (data.collector.enabled: false); enable it explicitly to fetch", err=True)
        raise typer.Exit(2)
    try:
        assets = assets_from_manifest(root / cfg.data.universe_manifest)

        def live():
            from .live_provider import build_live_provider

            return build_live_provider(cfg, assets)

        prov = make_provider(provider, source_dir=source_dir, license=license_text, live=live)
        out = output_snapshot if output_snapshot.is_absolute() else root / output_snapshot
        report = collect(assets, fx_pair=cfg.fx.pair_id, rates_symbol=cfg.cash.source_id, collector=col, provider=prov, start=start, end=end,
                         output=out, cache_root=root / col.cache_root, jitter_seed=cfg.reproducibility.random_streams.provider_jitter,
                         universe_source=root / cfg.data.universe_manifest)
    except (ValueError, OSError, SnapshotError, ProviderError) as e:
        typer.echo(f"data fetch failed: {e}", err=True)
        raise typer.Exit(2) from e
    c = report.cache
    typer.echo(f"data fetch: {report.status} ({len(report.items)} requests, cache hits {c['hits']}, misses {c['misses']})")
    for f in report.failures():
        typer.echo(f"FAIL {f['capability']}/{f['symbol']}: {f['kind']} after {f['attempts']} attempt(s): {f['error']}")
    if report.status == "success":
        typer.echo(f"snapshot {report.snapshot_id}")
    raise typer.Exit(0 if report.status == "success" else 1)


if __name__ == "__main__":
    app()
