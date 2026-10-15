"""Corporate-action reconciliation (spec 4.3, check ``INV-DATA-002``): three levels and an explicit halt scope.

1. Split ledger integrity: a split must scale a held quantity to a whole number of shares; the unexplained relative residual of the
   quantity x price identity must stay within ``split_relative_tolerance`` or the whole run halts before any later decision.
2. Provider total-return reconstruction: per asset, gross total return is rebuilt from raw prices and declared actions under
   ``PROVIDER_CONVENTION`` and compared with the provider's ``total_return_index``. A per-event residual above ``per_event_bps`` or a
   rolling cumulative residual above ``rolling_window_bps`` invalidates the asset from the first offending event.
3. Executable ledger versus adjusted research series: a reference buy-and-hold position is replayed with the ledger's own quantity and
   cash mechanics; the gap to the adjusted series is decomposed into named, analytically derived components and the remainder is the
   unexplained residual, judged by the same per-asset rule. On such a reference position the remainder equals the level-2 cumulative
   residual plus any ledger-mechanics residual (money rounding, entitlement, pay-date credit); trading costs and portfolio-level NAV are
   reconciled event by event elsewhere (INV-ACCT-001).

The fixture has no exclusion rule: every breach halts the entire run. Retrospective dropping is prohibited: an asset may be excluded only
by a rule frozen in the snapshot manifest, and never while it is held, has a pending order, is required by a benchmark, or its removal
breaches the universe minimum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .data import load_initial_holdings

PROVIDER_CONVENTION = {"dividends": "gross", "reinvestment": "ex_date_close", "splits": "ratio_adjusted_raw", "cash_amount": "per_share_held_at_ex_raw"}
ROLLING_WINDOW_RETURNS = 20  # the spec names a rolling window without a length (open-questions OQ-033)
REFERENCE_LOTS = 10_000  # reference buy-and-hold position (divisible by every supported split denominator used in practice)
SUPPORTED_ACTIONS = frozenset({"split", "cash_dividend"})
BPS = 1e4
EXCLUSION_RULE_FIELDS = ("event_types", "effective_cutoff_utc", "min_universe", "benchmark_consequence", "frozen_before_evaluation")


class ReconciliationHalt(RuntimeError):
    """A reconciliation breach that stops the whole run before any later decision (spec 4.3)."""


@dataclass(frozen=True)
class Breach:
    level: int
    instrument_id: str
    kind: str
    first_event_ts: str
    residual_bps: float | None
    limit_bps: float | None
    detail: str


@dataclass
class ReconciliationReport:
    breaches: list[Breach] = field(default_factory=list)
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    invalidated: dict[str, str] = field(default_factory=dict)  # asset -> first offending event timestamp
    halt: bool = False
    scope: str = "none"
    reason: str = ""

    def results(self) -> list[dict[str, object]]:
        """``validate_data``-style rows: one per level plus the halt-scope decision."""
        def row(level: int, name: str) -> dict[str, object]:
            bad = [b for b in self.breaches if b.level == level]
            return {"check": name, "status": "fail" if bad else "pass",
                    "detail": "; ".join(f"{b.instrument_id}:{b.kind}@{b.first_event_ts} {b.detail}" for b in bad) if bad else "within limits"}
        return [row(1, "ca_reconciliation_level1_split_ledger"), row(2, "ca_reconciliation_level2_total_return"),
                row(3, "ca_reconciliation_level3_executable_vs_adjusted"),
                {"check": "ca_reconciliation_halt_scope", "status": "fail" if self.halt else "pass", "detail": self.reason or "no breach"}]


# ------------------------------------------------------------------------------------------------------------- level 1
def split_quantity(quantity: int, num: int, den: int, tolerance: float) -> int:
    """Quantity after a ``num:den`` split; the residual of ``quantity x num == after x den`` must not exceed ``tolerance`` relatively.

    A whole-number result has zero residual. A fractional share is rounded toward zero (the declared lot rounding) only when the relative
    residual is within tolerance; otherwise the run halts.
    """
    if num <= 0 or den <= 0:
        raise ReconciliationHalt(f"split ratio {num}:{den} is not a positive ratio (level 1)")
    if quantity < 0:
        raise ReconciliationHalt(f"negative quantity {quantity} cannot be split (level 1)")
    after, remainder = divmod(quantity * num, den)
    if remainder and remainder / (quantity * num) > tolerance:
        raise ReconciliationHalt(f"fractional split quantity: {quantity} x {num}/{den} leaves {remainder}/{den} share, relative residual "
                                 f"{remainder / (quantity * num):.3e} > split_relative_tolerance {tolerance:.1e} (level 1)")
    return after


# ------------------------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class _Event:
    action_id: str
    instrument_id: str
    action_type: str
    ex_session: str
    ex_ts: pd.Timestamp
    pay_ts: pd.Timestamp | None
    num: int
    den: int
    cash_amount: float


def _events(actions: pd.DataFrame) -> list[_Event]:
    out = []
    for r in actions.itertuples(index=False):
        out.append(_Event(r.action_id, r.instrument_id, r.action_type, str(r.ex_session_date), pd.Timestamp(r.ex_timestamp_utc),
                          pd.Timestamp(r.pay_timestamp_utc) if isinstance(r.pay_timestamp_utc, str) else None,
                          int(r.ratio_num) if r.action_type == "split" else 1, int(r.ratio_den) if r.action_type == "split" else 1,
                          float(r.cash_amount) if r.action_type == "cash_dividend" else 0.0))
    return sorted(out, key=lambda e: (e.ex_ts, e.action_id))


def _aligned(raw: pd.DataFrame, tr: pd.DataFrame, instrument: str) -> pd.DataFrame:
    a = raw[raw.instrument_id == instrument][["event_timestamp_utc", "exchange_session", "value"]].rename(columns={"value": "raw"})
    b = tr[tr.instrument_id == instrument][["event_timestamp_utc", "value"]].rename(columns={"value": "tr"})
    m = a.merge(b, on="event_timestamp_utc", how="outer", indicator=True).sort_values("event_timestamp_utc").reset_index(drop=True)
    if (m["_merge"] != "both").any():
        raise ReconciliationHalt(f"{instrument}: raw and total-return observations do not share event timestamps (level 2)")
    for col in ("raw", "tr"):
        if not bool((m[col].map(math.isfinite) & (m[col] > 0)).all()):  # a corrupt series is a reported breach, never a crash
            raise ReconciliationHalt(f"{instrument}: {col} contains non-positive or non-finite values (level 2)")
    m["ts"] = pd.to_datetime(m.event_timestamp_utc, utc=True)
    return m


def _split_factor(ts: pd.Timestamp, splits: list[_Event]) -> float:
    f = 1.0
    for s in splits:
        if s.ex_ts <= ts:
            f *= s.num / s.den
    return f


# ------------------------------------------------------------------------------------------------------------- levels 2 and 3
def reconstruct_total_return(frame: pd.DataFrame, events: list[_Event], instrument: str) -> list[float]:
    """Gross total-return index of ``instrument`` rebuilt from raw prices and its declared actions (``PROVIDER_CONVENTION``), anchored at
    the first observation. Only the instrument's own actions apply."""
    splits = [e for e in events if e.action_type == "split" and e.instrument_id == instrument]
    divs = {e.ex_session: e for e in events if e.action_type == "cash_dividend" and e.instrument_id == instrument}
    out = [float(frame.tr.iloc[0])]
    prev_adj = float(frame.raw.iloc[0]) * _split_factor(frame.ts.iloc[0], splits)
    for i in range(1, len(frame)):
        s = _split_factor(frame.ts.iloc[i], splits)
        adj = float(frame.raw.iloc[i]) * s
        d = divs.get(str(frame.exchange_session.iloc[i]))
        out.append(out[-1] * (adj + (d.cash_amount * s if d else 0.0)) / prev_adj)
        prev_adj = adj
    return out


def _level2(frame: pd.DataFrame, events: list[_Event], inst: str, cfg_rec: Any) -> tuple[list[Breach], dict[str, float], list[float]]:
    recon = reconstruct_total_return(frame, events, inst)
    daily = [0.0] + [BPS * (math.log(recon[i] / recon[i - 1]) - math.log(frame.tr.iloc[i] / frame.tr.iloc[i - 1])) for i in range(1, len(frame))]
    breaches: list[Breach] = []
    sessions = {str(s): i for i, s in enumerate(frame.exchange_session)}
    max_event = 0.0
    for e in (x for x in events if x.instrument_id == inst):
        i = sessions.get(e.ex_session)
        if i is None:
            breaches.append(Breach(2, inst, "event_without_observation", str(e.ex_ts), None, None, f"{e.action_id}: ex session {e.ex_session} has no observation"))
            continue
        max_event = max(max_event, abs(daily[i]))
        if abs(daily[i]) > cfg_rec.per_event_bps:
            breaches.append(Breach(2, inst, "per_event", str(e.ex_ts), daily[i], cfg_rec.per_event_bps, f"{e.action_id} residual {daily[i]:+.3f} bps"))
    max_roll, first_roll = 0.0, None
    for i in range(1, len(frame)):
        window = sum(daily[max(1, i - ROLLING_WINDOW_RETURNS + 1): i + 1])
        max_roll = max(max_roll, abs(window))
        if first_roll is None and abs(window) > cfg_rec.rolling_window_bps:
            first_roll = (i, window)
    if first_roll:
        i, w = first_roll
        breaches.append(Breach(2, inst, "rolling_window", str(frame.ts.iloc[i]), w, cfg_rec.rolling_window_bps,
                               f"cumulative residual {w:+.3f} bps over {ROLLING_WINDOW_RETURNS} returns"))
    return breaches, {"max_per_event_bps": max_event, "max_rolling_bps": max_roll}, recon


def _level3(frame: pd.DataFrame, events: list[_Event], inst: str, recon: list[float], places: int, cfg_rec: Any) -> tuple[list[Breach], dict[str, float]]:
    splits = [e for e in events if e.action_type == "split" and e.instrument_id == inst]
    divs = [e for e in events if e.action_type == "cash_dividend" and e.instrument_id == inst]
    q = REFERENCE_LOTS
    quant = Decimal(1).scaleb(-places)
    entitlement: dict[str, int] = {}
    credited = Decimal(0)
    paid: set[str] = set()
    ledger_e, analytic_e, analytic_ex, analytic_after_ex = [], [], [], []
    for i in range(len(frame)):
        ts, price = frame.ts.iloc[i], frame.raw.iloc[i]
        for s in splits:  # ledger mechanics: the quantity is scaled by the level-1 rule at the ex event
            if s.ex_ts == ts:
                q = split_quantity(q, s.num, s.den, cfg_rec.split_relative_tolerance)
        for d in divs:
            if d.ex_ts == ts:
                entitlement[d.action_id] = q
            if d.pay_ts is not None and d.action_id in entitlement and d.action_id not in paid and d.pay_ts <= ts:
                credited += (Decimal(entitlement[d.action_id]) * Decimal(repr(d.cash_amount))).quantize(quant, rounding=ROUND_HALF_EVEN)
                paid.add(d.action_id)
        ledger_e.append(float(Decimal(q) * Decimal(repr(float(price))) + credited))
        s_i = _split_factor(ts, splits)  # analytic, independent of the ledger replay
        shares = REFERENCE_LOTS * s_i
        by_ex = sum(shares_at(d, REFERENCE_LOTS, splits) * d.cash_amount for d in divs if d.ex_ts <= ts)
        by_pay = sum(shares_at(d, REFERENCE_LOTS, splits) * d.cash_amount for d in divs if d.pay_ts is not None and d.pay_ts <= ts and d.ex_ts <= ts)
        analytic_e.append(shares * float(price) + by_pay)
        analytic_ex.append(shares * float(price) + by_ex)
        analytic_after_ex.append(by_ex - by_pay)
    ln = math.log
    worst = {"timing": 0.0, "non_reinvestment": 0.0, "lot_money_rounding": 0.0, "unexplained": 0.0}
    first_bad = None
    unexplained_series = []
    for i in range(1, len(frame)):
        total = ln(ledger_e[i] / ledger_e[0]) - ln(frame.tr.iloc[i] / frame.tr.iloc[0])
        timing = ln(analytic_e[i] / analytic_ex[i])
        non_reinv = (ln(analytic_ex[i] / analytic_ex[0]) - ln(recon[i] / recon[0]))
        rounding = (ln(ledger_e[i] / analytic_e[i]) - ln(ledger_e[0] / analytic_e[0]))
        explained = timing + non_reinv + rounding  # FX timing, withholding tax and costs are zero by construction (local currency, no costs)
        unexplained = BPS * (total - explained)
        unexplained_series.append(unexplained)
        worst["timing"] = max(worst["timing"], abs(BPS * timing))
        worst["non_reinvestment"] = max(worst["non_reinvestment"], abs(BPS * non_reinv))
        worst["lot_money_rounding"] = max(worst["lot_money_rounding"], abs(BPS * rounding))
        worst["unexplained"] = max(worst["unexplained"], abs(unexplained))
        if first_bad is None and abs(unexplained) > cfg_rec.rolling_window_bps:
            first_bad = (i, unexplained)
    breaches = []
    if first_bad:
        i, u = first_bad
        breaches.append(Breach(3, inst, "ledger_vs_adjusted", str(frame.ts.iloc[i]), u, cfg_rec.rolling_window_bps,
                               f"unexplained residual {u:+.3f} bps after decomposition"))
    return breaches, {f"level3_max_{k}_bps": v for k, v in worst.items()}


def shares_at(d: _Event, lots: int, splits: list[_Event]) -> float:
    """Shares entitled to ``d`` at its ex event (quantity after every split with ``ex <= d.ex``)."""
    return lots * _split_factor(d.ex_ts, splits)


# ------------------------------------------------------------------------------------------------------------- halt scope
@dataclass(frozen=True)
class ExclusionRule:
    """A per-asset exclusion rule frozen in the snapshot manifest before evaluation (spec 4.3 level 2)."""

    event_types: frozenset[str]
    effective_cutoff_utc: pd.Timestamp
    min_universe: int
    benchmark_consequence: str

    @staticmethod
    def parse(raw: dict[str, Any] | None) -> ExclusionRule | None:
        if raw is None:
            return None
        missing = [k for k in EXCLUSION_RULE_FIELDS if k not in raw]
        if missing:
            raise ReconciliationHalt(f"exclusion rule is missing {missing}: an incomplete rule cannot exclude an asset")
        if raw["frozen_before_evaluation"] is not True:
            raise ReconciliationHalt("exclusion rule is not marked frozen before evaluation: retrospective dropping is prohibited")
        return ExclusionRule(frozenset(raw["event_types"]), pd.Timestamp(raw["effective_cutoff_utc"]), int(raw["min_universe"]),
                             str(raw["benchmark_consequence"]))


def halt_scope(report: ReconciliationReport, *, fixture: bool, universe: set[str], universe_min: int, held: set[str], pending: set[str],
               benchmark_required: set[str], event_types: dict[str, str], rule: ExclusionRule | None) -> tuple[bool, str, str]:
    """Return ``(halt, scope, reason)``. Fixture: any breach halts the run. Research: only a valid frozen rule may exclude an asset."""
    if not report.breaches:
        return False, "none", ""
    if fixture:
        return True, "run", "the fixture has no exclusion rule: every reconciliation breach halts the entire run"
    if any(b.level == 1 for b in report.breaches):
        return True, "run", "split ledger integrity breach corrupts portfolio state (level 1)"
    bad = sorted(report.invalidated)
    if rule is None:
        return True, "run", f"no exclusion rule was frozen in the snapshot manifest; invalid assets {bad}"
    for a in bad:
        cutoff = pd.Timestamp(report.invalidated[a])
        why = None
        if event_types.get(a) not in rule.event_types:
            why = f"event type {event_types.get(a)!r} is not covered by the rule"
        elif cutoff < rule.effective_cutoff_utc:
            why = f"first offending event {cutoff} precedes the rule's effective cutoff {rule.effective_cutoff_utc}"
        elif a in held or a in pending:
            why = "the asset is held or has a pending order"
        elif a in benchmark_required:
            why = "the asset is required by a benchmark"
        if why:
            return True, "run", f"{a}: {why}"
    remaining = len(universe - set(bad))
    if remaining < max(universe_min, rule.min_universe):
        return True, "run", f"removing {bad} leaves {remaining} assets, below the universe minimum {max(universe_min, rule.min_universe)}"
    return False, "asset", f"assets {bad} are excluded under the frozen rule ({rule.benchmark_consequence})"


# ------------------------------------------------------------------------------------------------------------- entry points
def reconcile(cfg: Config, root: Path, *, rule: dict[str, Any] | None = None, pending: set[str] | None = None) -> ReconciliationReport:
    d = cfg.data
    root = Path(root)
    raw = pd.read_parquet(root / d.prices_raw)
    tr = pd.read_parquet(root / d.prices_total_return)
    actions = pd.read_parquet(root / d.corporate_actions)
    rep = ReconciliationReport()
    rec = cfg.accounting.corporate_action_reconciliation
    places = cfg.accounting.money_decimal_places
    unsupported = sorted(set(actions.action_type) - SUPPORTED_ACTIONS)
    for u in unsupported:
        first = actions[actions.action_type == u].iloc[0]
        rep.breaches.append(Breach(2, str(first.instrument_id), "unsupported_action", str(first.ex_timestamp_utc), None, None,
                                   f"{u} halts before the affected event (spec 4.3)"))
    events = [e for e in _events(actions) if e.action_type in SUPPORTED_ACTIONS]
    _cash, quantities, _asof = load_initial_holdings(cfg, root)
    for e in events:
        if e.action_type == "split":
            try:
                split_quantity(int(quantities.get(e.instrument_id, 0)), e.num, e.den, rec.split_relative_tolerance)
            except ReconciliationHalt as err:
                rep.breaches.append(Breach(1, e.instrument_id, "split_ledger", str(e.ex_ts), None, None, str(err)))
    for inst in sorted(set(raw.instrument_id)):
        try:
            frame = _aligned(raw, tr, inst)
            b2, diag, recon = _level2(frame, events, inst, rec)
            b3, diag3 = _level3(frame, events, inst, recon, places, rec)
        except ReconciliationHalt as err:
            rep.breaches.append(Breach(2, inst, "series", "", None, None, str(err)))
            continue
        rep.breaches += b2 + b3
        rep.diagnostics[inst] = {**diag, **diag3}
    for b in rep.breaches:
        if b.level >= 2 and b.first_event_ts and (b.instrument_id not in rep.invalidated or b.first_event_ts < rep.invalidated[b.instrument_id]):
            rep.invalidated[b.instrument_id] = b.first_event_ts
    assets = {a.asset_id for a in (cfg.universe.assets or [])}
    held = {a for a, q in quantities.items() if q}
    types = {e.instrument_id: e.action_type for e in events}
    rep.halt, rep.scope, rep.reason = halt_scope(
        rep, fixture=d.mode == "fixture", universe=assets, universe_min=cfg.universe.risky_asset_min_count, held=held,
        pending=pending or set(), benchmark_required={a.asset_id for a in (cfg.universe.assets or []) if a.benchmark_eligible},
        event_types=types, rule=ExclusionRule.parse(rule))
    return rep


def require_clean(cfg: Config, root: Path, **kw: Any) -> ReconciliationReport:
    """Run the reconciliation and raise ``ReconciliationHalt`` when the halt scope says the run must stop (before any decision)."""
    rep = reconcile(cfg, root, **kw)
    if rep.halt:
        first = rep.breaches[0]
        raise ReconciliationHalt(f"corporate-action reconciliation halts the run: {rep.reason} "
                                 f"[{first.instrument_id} level {first.level} {first.kind} @ {first.first_event_ts}: {first.detail}]")
    return rep
