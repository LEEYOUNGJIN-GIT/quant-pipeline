"""Market-specific order / scored-holding intervals and excluded observations for every block (spec 11.2, check ``INV-OOS-003``).

The model has no supervised forward label, so the label interval is an explicit null. The order interval runs from the block's first
decision cutoff to the last permitted execution close of its final decision; the scored holding interval is the set of NAV intervals
the block owns. Rows removed by a gap, purge or embargo, and the censored last decision, are listed with their cause.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .calendars import Calendars, RowMap, fmt, parse_delay
from .config import Config
from .splits import Block, SplitManifest


def _endpoints(cfg: Config, cals: Calendars, rm: RowMap, block: Block) -> dict[str, Any]:
    delay = parse_delay(cfg.availability.price_publication_delay)
    latency = parse_delay(cfg.decision_schedule.order_latency)
    first_cut = cals.close("XNYS", rm.eligible[block.start_pos - 1]) + delay
    fence = pd.Timestamp(block.fence_utc)
    out = {}
    for ex in dict.fromkeys(a.exchange for a in (cfg.universe.assets or [])):
        first_sess, _ = cals.next_close_after(ex, first_cut + latency)
        last_sess, _ = cals.next_close_after(ex, fence + latency)
        last = cals.cals[ex].session_offset(last_sess, cfg.decision_schedule.max_order_age_sessions)
        a, b = cals.close(ex, first_sess), cals.close(ex, last)
        out[ex] = {"first_execution_close_utc": fmt(a), "last_permitted_execution_close_utc": fmt(b),
                   "duration_hours": round((b - a).total_seconds() / 3600, 4)}
    return out


def _label(rm: RowMap, position: int) -> str:
    """The fixture decides at every model row (``D###``). Where decisions are a subset (research) a decision row is named by its decision id and any
    other model row is a plain row (``R####``): an excluded row must never be presented as a decision that did not exist."""
    if not rm.decisions:
        return f"D{position:03d}"
    by_position = {d.position: d.decision_id for d in rm.decisions}
    return by_position.get(position, f"R{position:04d}")


def _block_entry(cfg: Config, cals: Calendars, rm: RowMap, block: Block, excluded: list[dict[str, Any]]) -> dict[str, Any]:
    last = block.decision_positions[-1] if rm.decisions and block.decision_positions else block.end_pos
    return {"block": block.name, "decision_positions": [block.start_pos, block.end_pos],
            "label_interval": {"value": None, "reason": "no supervised forward label exists in this model"},
            "order_interval_utc": [block.first_decision_cutoff_utc, block.protected_end_utc],
            "scored_holding": {"valuation_sessions": list(block.valuation_sessions), "intervals": len(block.valuation_sessions),
                               "information_fence_utc": block.fence_utc},
            "market_endpoints": _endpoints(cfg, cals, rm, block), "excluded_observations": excluded,
            "censored_decision": {"id": _label(rm, last), "cause": "censored_at_block_end: its orders execute after the fence"}}


def _ids(rm: RowMap, positions: tuple[int, ...], cause: str) -> list[dict[str, Any]]:
    return [{"id": _label(rm, p), "position": p, "cause": cause} for p in positions]


def interval_report(cfg: Config, cals: Calendars, rm: RowMap, m: SplitManifest) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for p in m.passes:
        for f in p.inner:
            excl = _ids(rm, f.purge_positions, "purge_before_validation") + _ids(rm, f.gap_positions, "gap_after_validation")
            blocks.append({"scope": f"inner:{p.name}", **_block_entry(cfg, cals, rm, f.validation, excl)})
    for o in m.outer:
        blocks.append({"scope": "outer_oos", **_block_entry(cfg, cals, rm, o.test, [])})
    blocks.append({"scope": "holdout", **_block_entry(cfg, cals, rm, m.holdout, [])})

    findings = []
    for p in m.passes:  # within a pass the validation blocks must not share scored intervals or excluded rows with a block's own interior
        seen: dict[str, str] = {}
        for f in p.inner:
            for s in f.validation.valuation_sessions:
                if s in seen:
                    findings.append(f"{p.name}: scored interval {s} owned by {seen[s]} and {f.validation.name}")
                seen[s] = f.validation.name
            inside = set(f.validation.decision_positions)
            if inside & set(f.purge_positions) or inside & set(f.gap_positions):
                findings.append(f"{f.validation.name}: an excluded row lies inside the scored block")
    outer_sessions = [s for o in m.outer for s in o.test.valuation_sessions]
    if len(outer_sessions) != len(set(outer_sessions)):
        findings.append("outer test blocks share a scored interval")
    if set(outer_sessions) & set(m.holdout.valuation_sessions):
        findings.append("an outer scored interval is also a holdout interval")
    return {"schema": "oos-intervals-1", "split_manifest_hash": m.hash, "blocks": blocks, "overlap_findings": findings,
            "duplicate_scoring": bool(findings)}
