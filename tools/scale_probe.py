"""Full-scale timing probe for the research lane on a SYNTHETIC snapshot (engineering evidence only).

Builds a synthetic snapshot for the given window, runs the unreduced ``configs/research_8_12asset.yaml`` through ``ResearchPipeline`` for at most
``--stop-after`` durable checkpoints and prints the shape of the run, the runner and the time spent. It never reads provider data and writes only
below ``--work``. Intended for GitHub Actions and local sizing; nothing here is a research result.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--start", default="2015-01-02")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    from quant_pipeline.pipeline import StopForTest
    from quant_pipeline.research_pipeline import ResearchPipeline
    from support.synthetic_research import build_research_repo

    t0 = time.time()
    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)
    repo = args.work / "repo"
    repo.mkdir()
    root, snapshot = build_research_repo(repo / "r", start=args.start, end=args.end)
    (root / "configs").mkdir(exist_ok=True)
    shutil.copy(ROOT / "configs" / "research_8_12asset.yaml", root / "configs" / "research_8_12asset.yaml")
    t_build = time.time() - t0

    t1 = time.time()
    p = ResearchPipeline(root, root / "out", "scale", config="configs/research_8_12asset.yaml", stop_after=args.stop_after, allow_synthetic=True)
    m = p.manifest
    shape = {"window": [args.start, args.end], "model_rows": m.n_positions, "decision_rows": len(m.decision_positions), "outer_folds": len(m.outer),
             "holdout_rows": m.holdout.end_pos - m.holdout.start_pos + 1, "trials": len(p.ev.design.trials), "trial_space": p.ev.design.cardinality,
             "refits": len(m.refit_positions), "selection_passes": len(m.passes)}
    t_init = time.time() - t1
    info = {"python": platform.python_version(), "cpu_count": os.cpu_count(), "platform": platform.platform(), "snapshot": snapshot[:12]}
    print(json.dumps({"runner": info, "shape": shape, "seconds_build_snapshot": round(t_build, 1), "seconds_construct": round(t_init, 1)}, indent=1), flush=True)
    result = {"runner": info, "shape": shape, "seconds_build_snapshot": round(t_build, 1), "seconds_construct": round(t_init, 1)}
    if not args.plan_only:
        t2 = time.time()
        stopped = False
        try:
            p.run()
        except StopForTest:
            stopped = True
        result |= {"seconds_run": round(time.time() - t2, 1), "stopped_early": stopped, "checkpoints_written": p.written,
                   "computed": len(p.computed), "loaded": len(p.loaded)}
        print(json.dumps(result, indent=1), flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
