"""Split manifest (spec 11.1, 11.2): outer folds, inner selection passes, holdout, valuation ownership, event-time fences.

Positions are 1-indexed eligible decision rows. The manifest is computed from calendars and the profile only; it never
looks at prices or model output, so it cannot be tuned to results.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from .calendars import Calendars, RowMap, fmt, parse_delay, utc
from .config import Config
from .hashing import hash_json_obj


@dataclass(frozen=True)
class Block:
    name: str
    start_pos: int
    end_pos: int
    decision_positions: tuple[int, ...]
    valuation_sessions: tuple[str, ...]  # XNYS anchors whose NAV return the block owns (an interval ends at each)
    first_decision_cutoff_utc: str
    fence_utc: str  # information fence: valuation timestamp of the last owned anchor
    protected_start_utc: str
    protected_end_utc: str  # last permitted execution of the final decision's orders

    @property
    def n_decisions(self) -> int:
        return len(self.decision_positions)

    @property
    def n_valuation_intervals(self) -> int:
        return len(self.valuation_sessions)


@dataclass(frozen=True)
class InnerFold:
    index: int
    validation: Block
    gap_positions: tuple[int, ...]
    purge_positions: tuple[int, ...]
    embargo_positions: tuple[int, ...]
    path_end: int  # the isolated candidate path replays positions 1..path_end
    executable_decisions: int  # decisions whose orders can execute inside the fence (the last decision is censored)
    censored_decision: int


@dataclass(frozen=True)
class SelectionPass:
    name: str
    train_end: int
    inner: tuple[InnerFold, ...]
    refit_positions: tuple[int, ...]


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    train_end: int
    test: Block


@dataclass(frozen=True)
class SplitManifest:
    passes: tuple[SelectionPass, ...]
    outer: tuple[OuterFold, ...]
    holdout: Block
    refit_positions: tuple[int, ...]
    n_positions: int
    hash: str

    def path_endpoints(self) -> list[int]:
        return sorted({f.path_end for p in self.passes for f in p.inner})

    def outer_valuation_intervals(self) -> int:
        return sum(o.test.n_valuation_intervals for o in self.outer)


def _block(name: str, start: int, end: int, cfg: Config, cals: Calendars, rm: RowMap) -> Block:
    delay = parse_delay(cfg.availability.price_publication_delay)
    latency = parse_delay(cfg.decision_schedule.order_latency)
    first_s, last_s = rm.eligible[start - 1], rm.eligible[end - 1]
    anchors = tuple(s.strftime("%Y-%m-%d") for s in rm.anchors if first_s <= s <= last_s)
    fence = cals.close("XNYS", last_s) + delay
    exch = [a.exchange for a in (cfg.universe.assets or [])]
    tails = []
    for ex in dict.fromkeys(exch):
        sess, _ = cals.next_close_after(ex, fence + latency)
        last = cals.cals[ex].session_offset(sess, cfg.decision_schedule.max_order_age_sessions)
        tails.append(cals.close(ex, last))
    return Block(name, start, end, tuple(range(start, end + 1)), anchors, fmt(cals.close("XNYS", first_s) + delay), fmt(fence),
                 fmt(cals.close("XNYS", first_s) + delay), fmt(max(tails)))


def refit_schedule(cfg: Config, last_position: int) -> tuple[int, ...]:
    r = cfg.evaluation.continuous_refit
    first = 1 if r.start_at_first_eligible_decision else r.every_rows
    return tuple(range(first, last_position + 1, r.every_rows))


def build_split_manifest(cfg: Config, cals: Calendars, rm: RowMap) -> SplitManifest:
    ev, out = cfg.evaluation, cfg.evaluation.outer
    n = len(rm.eligible)
    hold_rows = cfg.final_holdout.rows
    if hold_rows is None:
        raise NotImplementedError("fractional holdouts belong to the research snapshot (Phase 5.7)")
    pre = n - hold_rows
    purge, embargo = ev.purge.minimum_rows, ev.embargo_rows
    folds = []
    for k in range(out.maximum_folds):
        train_end = out.initial_train_rows + k * out.step_rows
        if train_end + out.test_rows > pre:
            raise ValueError("outer test overlaps the holdout")
        folds.append(OuterFold(f"outer-{k + 1}", train_end, _block(f"outer-{k + 1}-test", train_end + 1, train_end + out.test_rows, cfg, cals, rm)))
    passes = []
    for name, t in [(f.fold_id, f.train_end) for f in folds] + [("final", pre)]:
        inner = []
        v, g = ev.inner.validation_rows, embargo
        for j in range(ev.inner.folds):
            end = t - (v + g) * (ev.inner.folds - 1 - j)
            start = end - v + 1
            if start - purge < ev.inner.minimum_train_rows - 0 and start - purge < 1:
                raise ValueError("inner validation block leaves no training history")
            block = _block(f"{name}-inner-{j + 1}", start, end, cfg, cals, rm)
            inner.append(InnerFold(j + 1, block, tuple(range(end + 1, end + 1 + g)) if j < ev.inner.folds - 1 else (),
                                   tuple(range(start - purge, start)), tuple(range(end + 1, end + 1 + g)), end,
                                   block.n_decisions - 1, end))
        passes.append(SelectionPass(name, t, tuple(inner), refit_schedule(cfg, t)))
    holdout = _block("holdout", pre + 1, n, cfg, cals, rm)
    body = {"passes": [[p.name, p.train_end, [[f.validation.start_pos, f.validation.end_pos, f.path_end, list(f.purge_positions),
                                              list(f.gap_positions), f.validation.valuation_sessions] for f in p.inner]] for p in passes],
            "outer": [[o.fold_id, o.train_end, o.test.start_pos, o.test.end_pos, list(o.test.valuation_sessions)] for o in folds],
            "holdout": [holdout.start_pos, holdout.end_pos, list(holdout.valuation_sessions)],
            "refits": list(refit_schedule(cfg, n))}
    m = SplitManifest(tuple(passes), tuple(folds), holdout, refit_schedule(cfg, n), n, hash_json_obj(body))
    validate_manifest(cfg, m)
    return m


def validate_manifest(cfg: Config, m: SplitManifest) -> None:
    """Row count alone is not evidence of OOS exercise: every block must meet its decision/valuation minima."""
    ev, out, fh = cfg.evaluation, cfg.evaluation.outer, cfg.final_holdout
    for p in m.passes:
        blocks = sorted((f.validation for f in p.inner), key=lambda b: b.start_pos)
        for a, b in itertools.pairwise(blocks):
            if a.end_pos >= b.start_pos:
                raise ValueError(f"{p.name}: validation blocks overlap")
        for f in p.inner:
            if f.executable_decisions < ev.inner.minimum_decisions_per_validation:
                raise ValueError(f"{p.name}: fewer than {ev.inner.minimum_decisions_per_validation} executable decisions")
            if f.validation.n_valuation_intervals < ev.inner.minimum_scored_nav_intervals_per_validation:
                raise ValueError(f"{p.name}: too few scored valuation intervals")
            if f.validation.end_pos > p.train_end:
                raise ValueError(f"{p.name}: validation extends past the training end")
    for o in m.outer:
        if o.test.n_decisions < out.minimum_decisions_per_test or o.test.n_valuation_intervals < out.minimum_scored_nav_intervals_per_test:
            raise ValueError(f"{o.fold_id}: test block below its minima")
        if o.test.start_pos != o.train_end + 1:
            raise ValueError(f"{o.fold_id}: test must start immediately after training")
    if m.holdout.n_decisions < fh.minimum_decisions or m.holdout.n_valuation_intervals < fh.minimum_scored_nav_intervals:
        raise ValueError("holdout below its minima")
    if cfg.valuation.expected_outer_oos_intervals is not None and m.outer_valuation_intervals() != cfg.valuation.expected_outer_oos_intervals:
        raise ValueError("outer OOS valuation intervals differ from the profile")
    if cfg.valuation.expected_holdout_intervals is not None and m.holdout.n_valuation_intervals != cfg.valuation.expected_holdout_intervals:
        raise ValueError("holdout valuation intervals differ from the profile")
    scored = [s for o in m.outer for s in o.test.valuation_sessions]
    if len(scored) != len(set(scored)) or set(scored) & set(m.holdout.valuation_sessions):
        raise ValueError("a valuation interval is owned twice or leaks into the holdout")


def censored_orders_note(block: Block) -> str:
    return f"orders of the final decision (position {block.end_pos}) execute after {block.fence_utc} and are censored_at_validation_end"


def purge_range_clash(refits: tuple[int, ...], inner: InnerFold) -> tuple[int, ...]:
    """Refit positions that fall inside the purge range of a validation block (they must be dropped for that fold)."""
    return tuple(r for r in refits if r in inner.purge_positions)


__all__ = [
    "Block",
    "InnerFold",
    "OuterFold",
    "SelectionPass",
    "SplitManifest",
    "build_split_manifest",
    "pd",
    "refit_schedule",
    "utc",
    "validate_manifest",
]
