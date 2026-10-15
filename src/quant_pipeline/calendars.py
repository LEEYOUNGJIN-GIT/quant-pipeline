"""Official exchange calendars, execution closes and row ownership (spec 5.1, 5.1.1, 6.2)."""
from __future__ import annotations

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
class RowMap:
    """Panel rows, complete-case rows, eligible (decision) rows and valuation owners are separate indexed sets."""

    anchors: pd.DatetimeIndex  # every XNYS anchor session in the decision window
    complete_case: tuple[pd.Timestamp, ...]
    warmup: tuple[pd.Timestamp, ...]
    eligible: tuple[pd.Timestamp, ...]
    owners: dict[str, tuple[pd.Timestamp, pd.Timestamp]]  # block -> (first, last) eligible session

    def position(self, session: pd.Timestamp) -> int:
        return self.eligible.index(session) + 1 if session in self.eligible else 0

    def valuation_owner(self, session: pd.Timestamp) -> str:
        return next((k for k, (a, b) in self.owners.items() if a <= session <= b), "")


def build_row_map(cfg: Config, cals: Calendars) -> tuple[RowMap, dict[str, pd.DatetimeIndex]]:
    fw = cfg.data.fixture_window
    if fw is None:
        raise NotImplementedError("row map for snapshot profiles arrives with the research snapshot (Phase 5)")
    start, end = fw.seed_observation_start_session, fw.decision_end_session
    sess = {n: cals.sessions(n, start, end) for n in ("XKRX", "XNYS")}
    both = sess["XNYS"].intersection(sess["XKRX"])
    complete_case = tuple(both[1:])  # the first common session has no preceding observed level
    warm = fw.warmup_rows
    eligible = complete_case[warm:]
    out = cfg.evaluation.outer
    hold = cfg.final_holdout.rows
    if hold is None:
        raise ValueError("fixture holdout needs an explicit row count")
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
