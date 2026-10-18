"""``hmm_pipeline_probe`` (spec 7.5): run the full EM -> export -> align -> filter -> boundary -> OOS pipeline on fixture
training rows in a fresh process and write everything an independent oracle needs to recompute it.

Run as ``python -m quant_pipeline.hmm_probe --out PATH``. Comparison with the oracle is done by
``tools/hmm_probe_oracle.py`` which does not import this package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calendars import Calendars, build_row_map
from .config import load_config
from .data import load_market_data
from .fitstate import HMM_FEATURE_ORDER, agent_context, decision_cutoff, fit_core, hmm_features, lineage_hashes
from .regimes import HMMParameters, run_hmm_pipeline

TRAIN_POSITION, OOS_POSITION = 40, 50  # outer fold 1: training end and the end of its first test block


def _arr(x: np.ndarray) -> list:
    return np.asarray(x, dtype=float).tolist()


def run_probe(root: Path, config_path: Path, out: Path) -> dict:
    cfg = load_config(config_path)
    cals = Calendars.load(["XKRX", "XNYS"])
    rm, sess = build_row_map(cfg, cals)
    md = load_market_data(cfg, root)
    hashes = lineage_hashes(cfg, root)
    fit = fit_core(cfg, md, cals, sess, fold_id="probe", refit_id="p0", hashes=hashes,
                   fit_end_utc=decision_cutoff(cals, cfg, rm.eligible[TRAIN_POSITION - 1]))
    ctx = agent_context(cfg, md, cals, sess, fit, decision_cutoff(cals, cfg, rm.eligible[OOS_POSITION - 1]))
    n_train = ctx.train_end_row + 1
    feats = hmm_features(fit, ctx.returns)
    train, oos = feats[:n_train], feats[n_train:]
    h = cfg.regime.hmm
    kw = {"n_states": h.n_states_candidates[0], "seed": cfg.reproducibility.random_streams.hmm_em,
          "max_iter": h.max_iterations, "tol": h.em_tolerance, "feature_order": HMM_FEATURE_ORDER}
    base = run_hmm_pipeline(train, oos, **kw)

    # non-identity permutation case: swap the raw states; alignment must restore the same aligned model
    raw = base.raw
    swap = list(range(raw.n_states))[::-1]
    swapped = HMMParameters(
        tuple(np.array(raw.start)[swap]), tuple(map(tuple, np.array(raw.transition)[np.ix_(swap, swap)])),
        tuple(map(tuple, np.array(raw.means)[swap])), tuple(map(tuple, np.array(raw.variances)[swap])),
        raw.feature_order, raw.converged, raw.fitter)
    permuted = run_hmm_pipeline(train, oos, **kw, raw_override=swapped)

    # future-row mutation: a changed OOS row may only affect posteriors from that row onward
    k = max(1, len(oos) // 2)
    mutated = oos.copy()
    mutated[k] += 5.0
    after = run_hmm_pipeline(train, mutated, **kw, raw_override=raw)

    payload = {
        "train": _arr(train), "oos": _arr(oos), "raw_params": json.loads(raw.to_json()),
        "aligned_params": json.loads(base.aligned.to_json()), "permutation": list(base.permutation),
        "start_fallback_stationary": base.start_fallback_stationary,
        "train_filtered": _arr(base.train.filtered), "train_predicted": _arr(base.train.predicted),
        "train_log_scale": _arr(base.train.log_scale),
        "boundary_posterior": _arr(base.boundary_posterior), "first_oos_predicted": _arr(base.first_oos_predicted),
        "oos_filtered": _arr(base.oos.filtered),
        "permuted_case": {"permutation": list(permuted.permutation), "oos_filtered": _arr(permuted.oos.filtered),
                          "boundary_posterior": _arr(permuted.boundary_posterior),
                          "aligned_params": json.loads(permuted.aligned.to_json())},
        "mutation_case": {"mutated_row": k, "oos_filtered": _arr(after.oos.filtered)},
        "em": {"converged": raw.converged, "fitter": raw.fitter, "seed": kw["seed"], "n_train_rows": int(n_train)},
    }
    out.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--config", default="configs/fixture_3asset.yaml", type=Path)
    args = ap.parse_args()
    run_probe(args.root.resolve(), args.root.resolve() / args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
