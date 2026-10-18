"""Phase 1 gate (spec 19): production kernels versus the independent oracle, plus the causality invariants.
Evidence class: ``fixture_path`` for oracle comparisons; the oracle files are ``independent_oracle``."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from quant_pipeline.calendars import Calendars, build_row_map, calendar_table, utc
from quant_pipeline.config import load_config
from quant_pipeline.data import load_initial_holdings, load_market_data
from quant_pipeline.event_time import PRIORITY, FillRequest, build_events
from quant_pipeline.panels import panel_tables
from quant_pipeline.replay import replay
from quant_pipeline.valuation import ValuationStaleBreach, max_stale_run, owned_interval_counts, valuation_table

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests" / "expected" / "fixture_3asset" / "oracle"


def _oracle_module():
    spec = importlib.util.spec_from_file_location("oracle_fixture", ROOT / "tools" / "oracle_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oracle_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


SCRIPTED = _oracle_module().SCRIPTED  # the same scripted fill set the oracle replayed (inputs only)
FILLS = [FillRequest(f"S-{d}-{a}", a, d, q) for d, a, q in SCRIPTED]


@pytest.fixture(scope="module")
def env():
    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml")
    cals = Calendars.load(["XKRX", "XNYS"])
    rm, sess = build_row_map(cfg, cals)
    return {"cfg": cfg, "cals": cals, "rm": rm, "sess": sess, "md": load_market_data(cfg, ROOT),
            "open": load_initial_holdings(cfg, ROOT)}


def run_replay(env, md=None, fills=FILLS, opening=None):
    cash, qty, asof = opening or env["open"]
    ev = build_events(env["cfg"], env["cals"], env["rm"], (md or env["md"]).actions, (md or env["md"]).rates, fills=fills)
    return replay(env["cfg"], md or env["md"], env["cals"], ev, cash, qty, utc(asof))


def test_calendar_table_equals_oracle(env):
    mine = calendar_table(env["cfg"], env["cals"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, pd.read_csv(ORACLE / "calendar_sessions.csv", keep_default_na=False), check_dtype=False)


def test_panel_equals_oracle(env):
    p = panel_tables(env["md"].total_return, env["cals"], env["sess"]["XNYS"], env["cfg"].availability.price_publication_delay)
    pd.testing.assert_frame_equal(p, pd.read_csv(ORACLE / "panel_daily_weekly.csv"), check_dtype=False, rtol=1e-12, atol=1e-12)


def test_event_queue_equals_oracle(env):
    ev = build_events(env["cfg"], env["cals"], env["rm"], env["md"].actions, env["md"].rates, include_slots=True)
    mine = pd.DataFrame([{"event_id": e.event_id, "ts": e.ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": e.kind,
                          "priority": PRIORITY[e.kind], "exchange": e.exchange, "asset_id": e.asset_id,
                          "order_id": e.order_id} for e in ev])
    pd.testing.assert_frame_equal(mine, pd.read_csv(ORACLE / "event_queue.csv", keep_default_na=False), check_dtype=False)


def test_ledger_replay_equals_oracle_exactly(env):
    got = run_replay(env).rows.astype(str)
    want = pd.read_csv(ORACLE / "ledger_projection.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(got, want)  # exact decimal strings, no tolerance


def test_every_fill_reconciles_self_financing(env):
    res = run_replay(env)
    assert len(res.reconciliation) == len(FILLS) and all(r["within_tolerance"] for r in res.reconciliation)
    assert res.ledger.cumulative_costs == sum(r["cost"] for r in res.reconciliation)


def test_panel_rows_are_invariant_to_future_observations(env, tmp_path):
    """INV-PANEL-002: changing observations unavailable by a cutoff cannot change rows at or before it."""
    cut = pd.Timestamp("2024-03-01T21:15:00Z")
    base = panel_tables(env["md"].total_return, env["cals"], env["sess"]["XNYS"], "PT15M")
    md2 = _perturbed_market(env, tmp_path, after=cut)
    pert = panel_tables(md2.total_return, env["cals"], env["sess"]["XNYS"], "PT15M")
    early = pd.to_datetime(base.cutoff_utc, utc=True) <= cut
    pd.testing.assert_frame_equal(base[early].reset_index(drop=True), pert[early].reset_index(drop=True))
    assert not base[~early].equals(pert[~early])  # the perturbation really changed later rows


def test_earlier_fills_are_invariant_to_later_mutation(env, tmp_path):
    """INV-ET-004 / spec 16.9: perturbing anything that happens after the KRX close leaves earlier postings unchanged."""
    cut = pd.Timestamp("2024-02-01T06:30:00Z")  # KRX closes: KRX_A and KRX_B are filled here
    base = run_replay(env).rows
    pert = run_replay(env, md=_perturbed_market(env, tmp_path, after=cut, by="event_timestamp_utc")).rows
    early = pd.to_datetime(base.ts, utc=True) <= cut
    pd.testing.assert_frame_equal(base[early].reset_index(drop=True), pert[early].reset_index(drop=True))
    later = base[~early].reset_index(drop=True)
    assert not later[["cash_krw", "nav_krw", "note"]].equals(pert[~early].reset_index(drop=True)[["cash_krw", "nav_krw", "note"]])


def _perturbed_market(env, tmp_path: Path, after: pd.Timestamp, by: str = "available_at_utc"):
    """Scale every observation whose ``by`` timestamp is after ``after``.

    ``available_at_utc`` is the right criterion for information used by decisions. For fills use
    ``event_timestamp_utc``: the executable close printed at the fill event is the fill price itself even
    though its publication lags by the delay (OQ-012).
    """
    root = tmp_path / "perturbed"
    shutil.copytree(ROOT / "data" / "fixtures", root / "data" / "fixtures")
    for name in ("prices_raw", "prices_total_return", "fx_krw_per_usd"):
        path = root / "data" / "fixtures" / "fixture_3asset" / f"{name}.parquet"
        df = pd.read_parquet(path)
        late = pd.to_datetime(df[by], utc=True) > after
        df.loc[late, "value"] = df.loc[late, "value"] * 1.07
        df.to_parquet(path, index=False)
    return load_market_data(env["cfg"], root)


def test_next_executable_close_is_strictly_after_every_decision(env):
    ev = build_events(env["cfg"], env["cals"], env["rm"], env["md"].actions, env["md"].rates, include_slots=True)
    cutoff = {f"D{i:03d}": env["cals"].close("XNYS", s) + pd.Timedelta(minutes=15)
              for i, s in enumerate(env["rm"].eligible, start=1)}
    slots = [e for e in ev if e.order_id.startswith("D") and e.kind.endswith("_fill")]
    assert len(slots) == 70 * 3 * 2
    for e in slots:
        c = cutoff[e.order_id.split("-")[0]]
        nxt = env["cals"].next_close_after(e.exchange, c)[1]
        assert e.ts == nxt and e.ts > c  # first close strictly after the decision, never the information close


def test_sequential_posting_later_nyse_proceeds_cannot_fund_earlier_krx_buy(env):
    opening = (Decimal("1000.00"), {"KRX_A": 0, "KRX_B": 0, "NYSE_C": 100}, pd.Timestamp("2024-01-31T21:15:00Z"))
    fills = [FillRequest("T-KRX-BUY", "KRX_A", "2024-02-01", 5), FillRequest("T-NYSE-SELL", "NYSE_C", "2024-02-01", -100)]
    with pytest.raises(ValueError, match="negative"):
        run_replay(env, fills=fills, opening=opening)
    funded = (Decimal("1000.00"), {"KRX_A": 0, "KRX_B": 0, "NYSE_C": 100}, pd.Timestamp("2024-01-31T21:15:00Z"))
    ok = run_replay(env, fills=[FillRequest("T-NYSE-SELL", "NYSE_C", "2024-02-01", -100)], opening=funded)
    assert Decimal(ok.rows.cash_krw.iloc[-1]) > 1000  # the sale is fine on its own


def test_same_timestamp_sells_post_before_buys(env):
    fills = [FillRequest("A-buy", "KRX_A", "2024-02-01", 10), FillRequest("B-buy", "KRX_B", "2024-02-01", 5),
             FillRequest("Z-sell", "KRX_A", "2024-03-20", -1000)]
    ev = build_events(env["cfg"], env["cals"], env["rm"], env["md"].actions, env["md"].rates, fills=[
        FillRequest("buyx", "KRX_B", "2024-03-20", 10), FillRequest("sellx", "KRX_A", "2024-03-20", -1)])
    same = [e for e in ev if e.ts == pd.Timestamp("2024-03-20T06:30:00Z")]
    assert [e.kind for e in same] == ["sell_fill", "buy_fill"]  # despite asset/order-id ordering, sells go first
    assert fills  # constructed above only to document intent; not replayed


def test_valuation_intervals_and_stale_limit(env):
    t = valuation_table(env["cfg"], env["cals"], env["rm"], env["sess"]["XKRX"])
    assert owned_interval_counts(t) == {"outer1": 11, "outer2": 11, "holdout": 12}
    assert 11 + 11 == env["cfg"].valuation.expected_outer_oos_intervals
    assert t.krx_stale_count.max() <= env["cfg"].valuation.max_krx_stale_sessions == 2
    assert t[t.valuation_stale].krx_stale_count.min() >= 1 and len(t) == 97


def test_stale_limit_breach_halts(env):
    cfg = env["cfg"]
    tight = cfg.model_copy(update={"valuation": cfg.valuation.model_copy(update={"max_krx_stale_sessions": 0})})
    with pytest.raises(ValuationStaleBreach):
        valuation_table(tight, env["cals"], env["rm"], env["sess"]["XKRX"])


def test_research_stale_limit_six_matches_calendar_history():
    cals = Calendars.load(["XKRX", "XNYS"])
    start = "2012-01-02"  # fixed; exchange_calendars supports only the last 20 years from today, so a start near that edge breaks on a given day (2006-10-16 did)
    assert max_stale_run(cals.sessions("XNYS", start, "2026-10-12"), cals.sessions("XKRX", start, "2026-10-12")) == 6
