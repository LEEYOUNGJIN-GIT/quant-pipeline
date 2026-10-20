"""Re-run the finishing stage of a research run with a probe on the optimizer, to explain an ``EngineHalt`` (aggregate numbers only).

``optimize`` is wrapped, not changed: every call is forwarded unchanged and when the result halts, the solver attempts, the size of each hard-constraint
violation of the marked holdings and of the previous target, and the constraint parameters are printed. No price, quantity or ledger value is printed;
portfolio weights are shown only as sums over the constraint groups. Nothing is relaxed and the halt still stops the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline import engine, research_pipeline
from quant_pipeline.optimizer import hard_constraint_violations, one_way_turnover

_orig = engine.optimize
_calls = {"n": 0}


def describe(inp, res, previous_target) -> dict:
    n = inp.n
    free = 1e9
    out = {
        "decision_call": _calls["n"],
        "result": {"branch": res.branch, "status": res.status, "halt_reason": res.halt_reason, "notes": list(res.notes or ())},
        "attempts": [{k: a.get(k) for k in ("attempt", "status", "prim_res", "dual_res", "max_violation", "accepted", "infeasibility_confirmed")}
                     for a in res.attempts],
        "marked_holdings_violations": {k: round(v, 8) for k, v in hard_constraint_violations(inp.w_minus, inp, free).items()},
        "marked_holdings": {"risky_gross": round(float(inp.w_minus[:n].sum()), 6), "cash": round(float(inp.w_minus[n]), 6),
                            "usd_exposure": round(float(inp.w_minus[:n][inp.usd_mask].sum()), 6),
                            "largest_weight_over_cap": round(float((inp.w_minus[:n] - inp.max_weights).max()), 6)},
        "limits": {"b_final": round(float(inp.b_final), 6), "cash_floor": inp.target_cash_floor, "usd_limit": inp.usd_limit,
                   "turnover_limit": inp.turnover_limit, "tradable_assets": int(inp.tradable.sum()), "assets": n,
                   "cap_sum": round(float(inp.max_weights.sum()), 6)},
    }
    if previous_target is not None:
        out["previous_target_violations"] = {k: round(v, 8) for k, v in hard_constraint_violations(previous_target, inp, free).items()}
        out["previous_target_turnover_to_marked"] = round(one_way_turnover(previous_target, inp.w_minus), 6)
    else:
        out["previous_target_violations"] = None
    return out


def probe(inp, cfg, *, previous_target=None, first_decision=False, **kw):
    _calls["n"] += 1
    res = _orig(inp, cfg, previous_target=previous_target, first_decision=first_decision, **kw)
    if res.halted or res.weights is None:
        print("HALT PROBE " + json.dumps(describe(inp, res, previous_target), indent=1, sort_keys=True, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
    return res


def main() -> int:
    engine.optimize = probe
    return research_pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())
