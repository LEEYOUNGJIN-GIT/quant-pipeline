"""Official exchange calendars, execution closes and row ownership (spec 5.1, 5.1.1, 6.2)."""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import exchange_calendars as xc
import pandas as pd

from .config import Config

DELAY_RE = {"PT15M": pd.Timedelta(minutes=15), "PT0S": pd.Timedelta(0)}


def parse_delay(iso: str) -> pd.Timedelta:
    if iso in DELAY_RE:
        return DELAY_RE[iso]
    return pd.Timedelta(iso)  # pandas understands ISO 8601 durations such as PT15M


def utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fmt(ts: pd.Timestamp) -> str:
    return utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Calendars:
    cals: dict[str, xc.ExchangeCalendar]

    @classmethod
    def load(cls, names: list[str]) -> Calendars:
        return cls({n: xc.get_calendar(n) for n in names})

    def sessions(self, name: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        return self.cals[name].sessions_in_range(start, end)

    def close(self, name: str, session: pd.Timestamp) -> pd.Timestamp:
        return utc(self.cals[name].session_close(session))

    def next_close_after(self, name: str, cutoff: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First official close of ``name`` strictly after ``cutoff`` (spec 6.2). Never the same instant."""
        day = utc(cutoff).tz_localize(None).normalize()
        for s in self.cals[name].sessions_in_range(day, day + pd.Timedelta(days=10)):
            close = self.close(name, s)
            if close > utc(cutoff):
                return s, close
        raise RuntimeError(f"no {name} close within 10 days after {cutoff}")


@dataclass(frozen=True)
class DecisionRow:
    """One scheduled decision (spec 5.1: decision rows are a separate indexed set that points to one fit)."""

    decision_id: str
    session: pd.Timestamp  # the XNYS anchor session of the decision cutoff
    position: int  # the last eligible complete-case position at or before the anchor: the model row the decision may use
    on_complete_case_row: bool  # False: the anchor is an XKRX-closed session, so the decision uses the previous model row


@dataclass(frozen=True)
class RowMap:
    """Panel rows, complete-case rows, eligible (model) rows, decision rows and valuation owners are separate indexed sets.

    ``eligible`` are the post-warmup complete-case sessions; a position is the 1-based index into it. ``decisions`` is empty when the
    profile decides at every eligible row (the fixture): then every position is a decision and no other relationship is stored.
    """

    anchors: pd.DatetimeIndex  # every XNYS anchor session in the decision window
    complete_case: tuple[pd.Timestamp, ...]
    warmup: tuple[pd.Timestamp, ...]
    eligible: tuple[pd.Timestamp, ...]
    owners: dict[str, tuple[pd.Timestamp, pd.Timestamp]]  # block -> (first, last) eligible session
    decisions: tuple[DecisionRow, ...] = ()

    def position(self, session: pd.Timestamp) -> int:
        return self.eligible.index(session) + 1 if session in self.eligible else 0

    def valuation_owner(self, session: pd.Timestamp) -> str:
        return next((k for k, (a, b) in self.owners.items() if a <= session <= b), "")

    @property
    def decision_rows(self) -> tuple[DecisionRow, ...]:
        """Every decision in order. The fixture decides at each eligible row, so its decision ``D001`` is model position 1, and so on."""
        if self.decisions:
            return self.decisions
        return tuple(DecisionRow(f"D{i:03d}", s, i, True) for i, s in enumerate(self.eligible, start=1))

    @property
    def decision_positions(self) -> tuple[int, ...]:
        return tuple(d.position for d in self.decision_rows)


def holdout_rows(cfg: Config, n_eligible: int) -> int:
    """Rows of the sealed tail: explicit for ``tail_rows``; ``max(minimum_rows, ceil(fraction * n))`` for ``tail_fraction`` (OQ-041)."""
    fh = cfg.final_holdout
    if fh.split_mode == "tail_rows":
        assert fh.rows is not None
        return fh.rows
    assert fh.fraction is not None
    return max(fh.minimum_rows or 0, math.ceil(fh.fraction * n_eligible))


def outer_leftover_rows(cfg: Config, n_eligible: int) -> int:
    """Rows between the last whole outer test and the holdout when folds are anchored at the start (OQ-048). Research moves them into the first fold's
    initial training span so the last outer test ends on the row before the holdout: no unmanaged carry, no skipped decision. The fixture has none."""
    if cfg.data.fixture_window is not None:
        return 0
    out = cfg.evaluation.outer
    pre = n_eligible - holdout_rows(cfg, n_eligible)
    return pre - (out.initial_train_rows + (outer_fold_count(cfg, n_eligible) - 1) * out.step_rows + out.test_rows)


def outer_fold_count(cfg: Config, n_eligible: int) -> int:
    """The fixture runs exactly ``maximum_folds``; research derives the count from the data and never truncates to fit a ceiling."""
    out = cfg.evaluation.outer
    if cfg.data.fixture_window is not None:
        return out.maximum_folds
    pre = n_eligible - holdout_rows(cfg, n_eligible)
    folds = (pre - out.initial_train_rows) // out.step_rows
    if folds < 1:
        raise ValueError(f"the snapshot has {n_eligible} eligible rows: {pre} before the holdout is fewer than {out.initial_train_rows} training rows "
                         f"plus one {out.test_rows}-row test block")
    if folds > out.maximum_folds:
        raise ValueError(f"the snapshot requires {folds} outer folds but evaluation.outer.maximum_folds is {out.maximum_folds}; "
                         "the plan fails instead of truncating time, folds or decisions")
    return folds


def research_warmup_rows(cfg: Config) -> int:
    """Rows needed before the first model row (OQ-040): the largest of the panel, covariance and drawdown look-backs."""
    return max(cfg.panel.minimum_daily_rows, cfg.covariance.lookback_rows, cfg.features.drawdown_window)


def monthly_decisions(anchors: pd.DatetimeIndex, eligible: tuple[pd.Timestamp, ...]) -> tuple[DecisionRow, ...]:
    """The first eligible canonical XNYS cutoff of each calendar month (spec 6.1); runtime gate failures advance within the month (OQ-040)."""
    cc = set(eligible)
    usable = [s for s in anchors if eligible[0] <= s <= eligible[-1]]
    out: list[DecisionRow] = []
    seen: set[str] = set()
    for s in usable:
        month = s.strftime("%Y%m")
        if month in seen:
            continue
        seen.add(month)
        out.append(DecisionRow(f"d{month}", s, bisect.bisect_right(eligible, s), s in cc))
    return tuple(out)


def window_from_manifest(manifest: dict) -> tuple[str, str]:
    req = manifest["request"]
    return req["start"], req["end"]


def build_row_map(cfg: Config, cals: Calendars, window: tuple[str, str] | None = None,
                  unobserved: frozenset[str] = frozenset()) -> tuple[RowMap, dict[str, pd.DatetimeIndex]]:
    fw = cfg.data.fixture_window
    if fw is None:
        if window is None:
            raise ValueError("a snapshot profile needs the collected window (the manifest's request start/end) to build its row map")
        return _research_row_map(cfg, cals, *window, unobserved=unobserved)
    start, end = fw.seed_observation_start_session, fw.decision_end_session
    sess = {n: cals.sessions(n, start, end) for n in ("XKRX", "XNYS")}
    both = sess["XNYS"].intersection(sess["XKRX"])
    complete_case = tuple(both[1:])  # the first common session has no preceding observed level
    warm = fw.warmup_rows
    eligible = complete_case[warm:]
    out = cfg.evaluation.outer
    hold = holdout_rows(cfg, len(eligible))
    bounds = {"outer1": (out.initial_train_rows + 1, out.initial_train_rows + out.test_rows),
              "outer2": (out.initial_train_rows + out.test_rows + 1, out.initial_train_rows + 2 * out.test_rows),
              "holdout": (len(eligible) - hold + 1, len(eligible))}
    owners = {k: (eligible[lo - 1], eligible[hi - 1]) for k, (lo, hi) in bounds.items()}
    anchors = sess["XNYS"][sess["XNYS"] >= pd.Timestamp(fw.decision_start_session)]
    rm = RowMap(anchors, complete_case, complete_case[:warm], eligible, owners)
    counts = (len(anchors), len(complete_case), len(rm.warmup), len(eligible))
    want = (fw.xnys_anchor_rows, fw.complete_case_return_rows, fw.warmup_rows, fw.model_eligible_rows)
    if counts != want:
        raise ValueError(f"fixture geometry mismatch: got {counts}, profile declares {want}")
    return rm, sess


def _research_row_map(cfg: Config, cals: Calendars, start: str, end: str,
                      unobserved: frozenset[str] = frozenset()) -> tuple[RowMap, dict[str, pd.DatetimeIndex]]:
    sess = {n: cals.sessions(n, start, end) for n in ("XKRX", "XNYS")}
    both = sess["XNYS"].intersection(sess["XKRX"])
    # a common session on which some asset has no observation is not a complete-case row (spec 5.1 rule 6); empty for gap-free data
    complete_case = tuple(x for x in both[1:] if x.strftime("%Y-%m-%d") not in unobserved)
    warm = research_warmup_rows(cfg)
    eligible = complete_case[warm:]
    if not eligible:
        raise ValueError(f"{len(complete_case)} complete-case rows do not exceed the {warm}-row warmup")
    out, n = cfg.evaluation.outer, len(eligible)
    folds, hold = outer_fold_count(cfg, n), holdout_rows(cfg, n)
    lead = outer_leftover_rows(cfg, n)
    bounds = {f"outer{k}": (lead + out.initial_train_rows + (k - 1) * out.step_rows + 1, lead + out.initial_train_rows + (k - 1) * out.step_rows + out.test_rows)
              for k in range(1, folds + 1)}
    bounds["holdout"] = (n - hold + 1, n)
    owners = {k: (eligible[lo - 1], eligible[hi - 1]) for k, (lo, hi) in bounds.items()}
    rm = RowMap(sess["XNYS"], complete_case, complete_case[:warm], eligible, owners, monthly_decisions(sess["XNYS"], eligible))
    return rm, sess


def calendar_table(cfg: Config, cals: Calendars) -> pd.DataFrame:
    rm, sess = build_row_map(cfg, cals)
    cc, elig, warm = set(rm.complete_case), set(rm.eligible), set(rm.warmup)
    rows = []
    for s in sess["XNYS"]:
        is_anchor = s in set(rm.anchors)
        rows.append({
            "calendar": "XNYS", "session_date": s.strftime("%Y-%m-%d"), "close_utc": fmt(cals.close("XNYS", s)),
            "is_anchor": is_anchor, "also_xkrx_open": s in set(sess["XKRX"]), "complete_case": s in cc,
            "eligible_position": rm.position(s),
            "role": "warmup" if s in warm else "eligible" if s in elig else "none",
            "valuation_owner": rm.valuation_owner(s) if is_anchor else "",
        })
    for s in sess["XKRX"]:
        rows.append({"calendar": "XKRX", "session_date": s.strftime("%Y-%m-%d"), "close_utc": fmt(cals.close("XKRX", s)),
                     "is_anchor": False, "also_xkrx_open": True, "complete_case": False, "eligible_position": 0,
                     "role": "none", "valuation_owner": ""})
    return pd.DataFrame(rows)
