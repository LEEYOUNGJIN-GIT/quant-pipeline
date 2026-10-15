"""Graph-derived compute plan and the after-plan gate (spec 11.8, 13, 18).

Counts come from the resolved execution graph (folds, refits, trials, cost scenarios, calibration cutoffs), never from constants.
Every unit is reported twice: the **no-sharing upper bound** and the **fingerprint-deduplicated** planned count. The profile's
``compute_budget`` ceilings were declared before any plan existed and are compared with the deduplicated counts; they are never
adjusted to fit a plan.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .hashing import hash_json_obj
from .splits import SplitManifest
from .trials import TrialDesign

MULTIPLIERS = (0.0, 1.0, 2.0)
PLAN_SCHEMA = "compute-plan-1"
CAP_FIELDS = {
    "model_fit_unit": "max_model_fit_units", "calibration_unit": "max_calibration_units",
    "rolling_kernel_evaluation": "max_rolling_kernel_evaluations", "allocation_optimizer_solve": "max_allocation_optimizer_solves",
    "auxiliary_optimizer_solve": "max_auxiliary_optimizer_solves", "ledger_replay": "max_ledger_replays",
    "bootstrap_replicate": "max_bootstrap_replicates", "checkpoint": "max_checkpoints",
}
BYTES_PER_DECISION_ARTIFACTS = 6_000_000  # crude, recorded as an assumption: plans + ledger rows + checkpoints per decision path
BYTES_PER_FIT = 2_000_000


def refits_upto(refits: tuple[int, ...], last: int) -> list[int]:
    return [r for r in refits if r <= last]


def refit_of(refits: tuple[int, ...], position: int) -> int:
    return max(r for r in refits if r <= position)


def build_compute_plan(cfg: Config, manifest: SplitManifest, design: TrialDesign, hashes: dict[str, str]) -> dict[str, Any]:
    refits = manifest.refit_positions
    n = manifest.n_positions
    n_trials = len(design.trials)
    gammas = len({json.dumps(t["parameters"]["agents.defensive.overlay_gamma_candidates"]) for t in design.trials})
    inner = [(p.name, f.index, f.path_end) for p in manifest.passes for f in p.inner]
    committed = [(o.fold_id, o.test.start_pos, o.test.end_pos) for o in manifest.outer] + [("holdout", manifest.holdout.start_pos, manifest.holdout.end_pos)]
    committed_decisions = sum(e - s + 1 for _, s, e in committed)
    mults = len(cfg.costs.sensitivity_multipliers)
    attempts = cfg.compute_budget.max_solver_attempts_per_problem

    # ---- model_fit_unit: distinct (information cutoff, refit policy) -------------------------------------------------------
    fit_dedup = len(refits) + (1 if cfg.regime.hmm_pipeline_probe.enabled else 0)
    fit_nosharing = (sum(len(refits_upto(refits, e)) for _, _, e in inner) * n_trials
                     + sum(len([r for r in refits if s <= r <= e]) for _, s, e in committed) * mults
                     + (1 if cfg.regime.hmm_pipeline_probe.enabled else 0))

    # ---- calibration_unit: one per (refit, target-producing fingerprint) -------------------------------------------------
    calib_dedup = len(refits) * gammas
    calib_nosharing = (sum(len(refits_upto(refits, e)) for _, _, e in inner) * n_trials
                       + sum(len([r for r in refits if s <= r <= e]) for _, s, e in committed) * mults)

    # ---- calibration decisions (each needs one rolling-PCA evaluation, agents and an auxiliary min-variance solve) ------------
    cal_contexts = {(r, u) for r in refits for u in range(1, r + 1)}
    decision_contexts = {(refit_of(refits, p), p) for p in range(1, n + 1)}
    contexts = cal_contexts | decision_contexts
    aux_dedup = len(contexts)  # Diversification does not depend on gamma/lambda: one solve per (fit, position) context
    aux_nosharing = (sum(sum(r for r in refits_upto(refits, e)) + e for _, _, e in inner) * n_trials
                     + sum(sum(r for r in refits if s <= r <= e) + (e - s + 1) for _, s, e in committed) * mults)
    rolling_dedup = n  # the rolling PCA share depends only on returns up to a cutoff: one evaluation per distinct cutoff
    rolling_nosharing = aux_nosharing

    # ---- allocation solves (trial-sensitive: never shared across unequal trial hashes) -----------------------------------------
    alloc_dedup = n_trials * max(e for _, _, e in inner) + committed_decisions * mults
    alloc_nosharing = n_trials * sum(e for _, _, e in inner) + committed_decisions * mults

    # ---- ledger replays ---------------------------------------------------------------------------------------------------
    n_bench = len(cfg.benchmarks.definitions)
    replays_dedup = n_trials + len(committed) * mults * (1 + n_bench)
    replays_nosharing = len(inner) * n_trials + len(committed) * mults * (1 + n_bench)

    n_checkpoints = 1 + n_trials + 1 + (len(manifest.outer)) * mults + 1
    counts = {
        "model_fit_unit": {"deduplicated": fit_dedup, "no_sharing_upper_bound": fit_nosharing, "cap": cfg.compute_budget.max_model_fit_units},
        "calibration_unit": {"deduplicated": calib_dedup, "no_sharing_upper_bound": calib_nosharing, "cap": cfg.compute_budget.max_calibration_units},
        "rolling_kernel_evaluation": {"deduplicated": rolling_dedup, "no_sharing_upper_bound": rolling_nosharing,
                                      "cap": cfg.compute_budget.max_rolling_kernel_evaluations},
        "allocation_optimizer_solve": {"deduplicated": alloc_dedup, "no_sharing_upper_bound": alloc_nosharing,
                                       "cap": cfg.compute_budget.max_allocation_optimizer_solves},
        "auxiliary_optimizer_solve": {"deduplicated": aux_dedup, "no_sharing_upper_bound": aux_nosharing,
                                      "cap": cfg.compute_budget.max_auxiliary_optimizer_solves},
        "ledger_replay": {"deduplicated": replays_dedup, "no_sharing_upper_bound": replays_nosharing, "cap": cfg.compute_budget.max_ledger_replays},
        "bootstrap_replicate": {"deduplicated": cfg.bootstrap.replicates, "no_sharing_upper_bound": cfg.bootstrap.replicates,
                                "cap": cfg.compute_budget.max_bootstrap_replicates},
        "checkpoint": {"deduplicated": n_checkpoints, "no_sharing_upper_bound": n_checkpoints, "cap": cfg.compute_budget.max_checkpoints},
    }
    total_solves = {
        "deduplicated_problems": alloc_dedup + aux_dedup,
        "deduplicated_solver_attempts_upper_bound": (alloc_dedup + aux_dedup) * attempts,
        "no_sharing_problems": alloc_nosharing + aux_nosharing,
        "no_sharing_solver_attempts_upper_bound": (alloc_nosharing + aux_nosharing) * attempts,
        "cap": cfg.compute_budget.max_total_optimizer_solves, "max_attempts_per_problem": attempts,
    }
    graph = {
        "refit_positions": list(refits), "eligible_positions": n, "candidate_cardinality": design.cardinality,
        "trials_planned": n_trials, "attempted_sobol_bound": design.max_attempted_points, "sobol_points_used": len(design.points),
        "cost_multipliers": list(cfg.costs.sensitivity_multipliers), "distinct_gamma_values": gammas,
        "inner_paths": [{"pass": a, "inner_fold": b, "path_end": c} for a, b, c in inner],
        "committed_blocks": [{"block": a, "start": b, "end": c} for a, b, c in committed], "committed_decisions": committed_decisions,
        "split_manifest_hash": manifest.hash, "trial_design_hash": design.design_hash,
        "bootstrap_shape": [cfg.bootstrap.replicates, cfg.bootstrap.expected_oos_rows or 0],
        "calibration_contexts": len(cal_contexts), "decision_contexts": len(decision_contexts), "distinct_contexts": len(contexts),
    }
    estimates = {
        "assumptions": "crude; recorded so a reviewer can contest them",
        "peak_memory_gib": round((n * 80_000 + len(refits) * BYTES_PER_FIT * 4) / 2**30 + 0.5, 3),
        "disk_gib": round((n_checkpoints * (committed_decisions + n) * BYTES_PER_DECISION_ARTIFACTS / 50) / 2**30 + 0.1, 3),
    }
    body = {"schema": PLAN_SCHEMA, "profile": cfg.profile.name, "counts": counts, "total_optimizer_solves": total_solves, "graph": graph,
            "estimates": estimates, "operation_caps_from_plan": False,
            "lineage": {k: hashes[k] for k in ("spec_hash", "base_profile_hash", "data_manifest_hash", "code_hash")}}
    body["dependency_graph_hash"] = hash_json_obj(graph)
    body["plan_hash"] = hash_json_obj(body)
    return body


def write_plan(out_root: Path, run_id: str, plan: dict[str, Any]) -> Path:
    path = Path(out_root) / run_id / "compute_plan.json"
    if path.exists():
        raise FileExistsError(f"compute_plan.json is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def host_capacity() -> dict[str, float]:
    cpu = os.cpu_count() or 1
    ram = float("nan")
    try:
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        if os.name == "nt":
            s = MemStatus()
            s.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            ram = s.ullAvailPhys / 2**30
        else:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, AttributeError, ValueError):  # capacity probing is best effort and reported as NaN when unavailable
        ram = float("nan")
    return {"logical_cpu": float(cpu), "available_ram_gib": ram, "free_disk_gib": shutil.disk_usage(Path.cwd()).free / 2**30}


def microbenchmark() -> dict[str, float]:
    """Measure one tiny convex solve so the wall estimate rests on a recorded number, not an assumption."""
    import cvxpy as cp

    n = 4
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, n))
    q = a @ a.T / n + np.eye(n)
    t0 = time.perf_counter()
    for _ in range(5):
        w = cp.Variable(n)
        cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(q))), [cp.sum(w) == 1, w >= 0]).solve(solver=cp.OSQP)
    return {"seconds_per_small_solve": (time.perf_counter() - t0) / 5}


def doctor_after_plan(cfg: Config, plan: dict[str, Any], plan_path: Path, out_root: Path, run_id: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def chk(name: str, ok: bool, detail: str, blocking: bool = True) -> None:
        checks.append({"check": name, "status": "pass" if ok else "fail", "blocking": blocking, "detail": detail})

    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    chk("plan_hash_matches_content", hash_json_obj(body) == plan["plan_hash"], plan["plan_hash"][:19])
    chk("dependency_graph_hash_matches", hash_json_obj(plan["graph"]) == plan["dependency_graph_hash"], plan["dependency_graph_hash"][:19])
    g = plan["graph"]
    chk("candidate_cardinality_within_cap", g["trials_planned"] <= cfg.evaluation.max_parameter_trials,
        f"{g['trials_planned']} trials planned, cap {cfg.evaluation.max_parameter_trials}, cardinality {g['candidate_cardinality']}")
    chk("sobol_points_within_attempt_bound", g["sobol_points_used"] <= g["attempted_sobol_bound"],
        f"{g['sobol_points_used']} points used of at most {g['attempted_sobol_bound']}")
    chk("bootstrap_shape", g["bootstrap_shape"] == [cfg.bootstrap.replicates, cfg.bootstrap.expected_oos_rows or 0], str(g["bootstrap_shape"]))
    chk("caps_not_copied_from_plan", plan["operation_caps_from_plan"] is False, "operation_caps_from_plan is false")
    for unit, c in plan["counts"].items():
        cap = getattr(cfg.compute_budget, CAP_FIELDS[unit])  # the ceiling comes from the CURRENT profile, never from the plan file
        chk(f"plan_records_profile_cap[{unit}]", c["cap"] == cap, f"plan recorded {c['cap']}, profile declares {cap}")
        # the bootstrap replicate count is a configured input equal to its ceiling by definition, not a graph-derived count
        independent = cap is not None and cap > 0 and (unit == "bootstrap_replicate"
                                                     or (cap != c["deduplicated"] and cap != c["no_sharing_upper_bound"]))
        chk(f"cap_is_independent[{unit}]", independent, f"cap {cap} differs from planned {c['deduplicated']} and bound {c['no_sharing_upper_bound']}")
        chk(f"within_cap[{unit}]", c["deduplicated"] <= cap,
            f"deduplicated {c['deduplicated']} vs cap {cap} (no-sharing upper bound {c['no_sharing_upper_bound']})")
    t = plan["total_optimizer_solves"]
    total_cap = cfg.compute_budget.max_total_optimizer_solves
    chk("plan_records_profile_cap[total_optimizer_solves]", t["cap"] == total_cap, f"plan recorded {t['cap']}, profile declares {total_cap}")
    chk("within_cap[total_optimizer_solves]", t["deduplicated_solver_attempts_upper_bound"] <= total_cap,
        f"{t['deduplicated_solver_attempts_upper_bound']} attempts (upper bound) vs cap {total_cap}; no-sharing {t['no_sharing_solver_attempts_upper_bound']}")
    host = host_capacity()
    cb = cfg.compute_budget
    chk("host_cpu", host["logical_cpu"] >= cb.logical_cpu_cores, f"{host['logical_cpu']:.0f} available vs {cb.logical_cpu_cores} required")
    chk("host_ram", host["available_ram_gib"] != host["available_ram_gib"] or host["available_ram_gib"] >= cb.memory_gib,
        f"{host['available_ram_gib']:.1f} GiB available vs {cb.memory_gib} configured", blocking=host["available_ram_gib"] == host["available_ram_gib"])
    chk("host_disk", host["free_disk_gib"] >= max(cb.disk_gib, plan["estimates"]["disk_gib"]),
        f"{host['free_disk_gib']:.1f} GiB free vs {cb.disk_gib} configured / {plan['estimates']['disk_gib']} estimated")
    bench = microbenchmark()
    est_minutes = (t["deduplicated_solver_attempts_upper_bound"] * bench["seconds_per_small_solve"] * 10) / 60  # x10: agents, plans, ledger
    chk("wall_estimate_within_budget", est_minutes <= cb.total_wall_clock_minutes,
        f"{est_minutes:.1f} min estimated (microbenchmark {bench['seconds_per_small_solve']:.4f} s/solve, x10 overhead assumption) "
        f"vs {cb.total_wall_clock_minutes} min", blocking=False)
    failed = [c for c in checks if c["status"] == "fail" and c["blocking"]]
    report = {"doctor": "after_plan", "run_id": run_id, "plan_hash": plan["plan_hash"], "status": "fail" if failed else "pass",
              "failed": [c["check"] for c in failed], "checks": checks, "host": host, "microbenchmark": bench,
              "note": "caps are independent ceilings declared before this plan; a failure is reported, never adapted away"}
    out = Path(out_root) / run_id / "doctor.after_plan.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report
