#!/usr/bin/env python3
"""Independent NumPy oracle for ``hmm_pipeline_probe`` (spec 7.5). MUST NOT import ``quant_pipeline``.

Recomputes, from the probe's serialized parameters and observations only: the volatility/mean/index alignment
permutation, the train-window forward filter (scaled, log-space, written independently of the production code), the
boundary posterior, the first OOS prediction and the OOS filter, and checks the permuted and mutated-row cases.
Exit code 0 means every posterior agrees to 1e-10; the report is printed as JSON.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

TOL = 1e-10


def lse(x: np.ndarray, axis: int = 0) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def loglik(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    out = np.zeros((y.shape[0], mean.shape[0]))
    for t in range(y.shape[0]):
        for s in range(mean.shape[0]):
            out[t, s] = -0.5 * sum((y[t, d] - mean[s, d]) ** 2 / var[s, d] + math.log(2 * math.pi * var[s, d])
                                   for d in range(y.shape[1]))
    return out


def forward(start, trans, mean, var, y, prior=None):
    """Return (predicted, filtered). ``prior=None`` starts from ``start``; otherwise the first row is predicted from it."""
    ll = loglik(y, mean, var)
    k = len(start)
    pred, filt = np.zeros((len(y), k)), np.zeros((len(y), k))
    for t in range(len(y)):
        if t == 0 and prior is None:
            lp = np.log(start)
        else:
            before = prior if t == 0 else filt[t - 1]
            lp = np.array([lse(np.log(before) + np.log(trans[:, c])) for c in range(k)])
        pred[t] = np.exp(lp - lse(lp))
        joint = lp + ll[t]
        filt[t] = np.exp(joint - lse(joint))
    return pred, filt


def align(raw):
    vol = np.sqrt(np.array(raw["variances"]).mean(axis=1))
    mean = np.array(raw["means"]).mean(axis=1)
    perm = sorted(range(len(vol)), key=lambda i: (vol[i], mean[i], i))
    s, a = np.array(raw["start"])[perm], np.array(raw["transition"])[np.ix_(perm, perm)]
    return perm, s, a, np.array(raw["means"])[perm], np.array(raw["variances"])[perm]


def main(path: str) -> int:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    train, oos = np.array(d["train"]), np.array(d["oos"])
    report, ok = {}, True

    def check(name: str, a, b) -> None:
        nonlocal ok
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        report[name] = err
        ok &= err <= TOL

    perm, s, a, m, v = align(d["raw_params"])
    report["permutation_matches"] = perm == d["permutation"]
    ok &= report["permutation_matches"]
    check("aligned_means", m, d["aligned_params"]["means"])
    check("aligned_transition", a, d["aligned_params"]["transition"])

    pred, filt = forward(s, a, m, v, train)
    check("train_filtered", filt, d["train_filtered"])
    check("train_predicted", pred, d["train_predicted"])
    boundary = filt[-1]
    check("boundary_posterior", boundary, d["boundary_posterior"])
    p2, f2 = forward(s, a, m, v, oos, prior=boundary)
    check("first_oos_predicted", p2[0], d["first_oos_predicted"])
    check("first_oos_equals_boundary_times_transition", boundary @ a, d["first_oos_predicted"])
    check("oos_filtered", f2, d["oos_filtered"])
    stationary = np.linalg.eig(a.T)[1][:, np.argmin(np.abs(np.linalg.eig(a.T)[0] - 1))].real
    stationary = stationary / stationary.sum()
    report["stationary_reset_would_differ"] = bool(np.abs(stationary @ a - p2[0]).max() > 1e-6)

    pc = d["permuted_case"]
    ident = list(range(len(pc["permutation"])))
    # the two cases are mirror images, so exactly one of them exercises a non-identity relabeling
    report["base_permutation"], report["permuted_case_permutation"] = d["permutation"], pc["permutation"]
    report["nonidentity_permutation_exercised"] = d["permutation"] != ident or pc["permutation"] != ident
    report["mirror_cases_use_different_permutations"] = d["permutation"] != pc["permutation"]
    ok &= report["nonidentity_permutation_exercised"] and report["mirror_cases_use_different_permutations"]
    check("permuted_case_oos_equals_base", pc["oos_filtered"], d["oos_filtered"])
    check("permuted_case_boundary_equals_base", pc["boundary_posterior"], d["boundary_posterior"])

    k = d["mutation_case"]["mutated_row"]
    mut = oos.copy()
    mut[k] += 5.0
    _, fm = forward(s, a, m, v, mut, prior=boundary)
    check("mutation_case_matches_independent_recompute", fm, d["mutation_case"]["oos_filtered"])
    check("rows_before_mutation_unchanged", np.array(d["mutation_case"]["oos_filtered"])[:k], np.array(d["oos_filtered"])[:k])
    report["rows_from_mutation_changed"] = bool(
        np.abs(np.array(d["mutation_case"]["oos_filtered"])[k:] - np.array(d["oos_filtered"])[k:]).max() > 1e-6)
    ok &= report["rows_from_mutation_changed"]

    report["status"] = "pass" if ok else "fail"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
