"""Oracle checks. Evidence classes: oracle files are ``independent_oracle`` (deterministic kernels only).
Cross-checks against production kernels prove agreement of two independent implementations, nothing more."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.accounting import CostRates, Fill, Ledger
from quant_pipeline.regimes import filter_window

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "tests" / "expected" / "fixture_3asset" / "oracle"
ORACLE_SRC = ROOT / "tools" / "oracle_fixture.py"


@pytest.fixture(scope="module")
def oracle():
    spec = importlib.util.spec_from_file_location("oracle_fixture", ORACLE_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oracle_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def test_oracle_does_not_import_the_package():
    tree = ORACLE_SRC.read_text(encoding="utf-8")
    body = tree.split('"""', 2)[2]
    assert "import quant_pipeline" not in body and "from quant_pipeline" not in body


def test_checked_in_oracle_files_match_a_fresh_recomputation(oracle):
    """Recomputing never writes; it must reproduce the frozen expected files byte for byte."""
    fresh = oracle.build_all()
    manifest = json.loads((ORACLE_DIR / "oracle_manifest.json").read_text(encoding="utf-8"))
    assert set(fresh) == set(manifest["output_hashes"])
    for name, content in fresh.items():
        assert (ORACLE_DIR / name).read_bytes() == content, name
        assert manifest["output_hashes"][name] == _sha(content)
    assert manifest["oracle_source_hash"] == _sha(ORACLE_SRC.read_bytes())
    inp = json.loads((ROOT / "data/fixtures/fixture_3asset/manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_manifest_hash"] == oracle.sha(__import__("rfc8785").dumps(inp))
    assert manifest["evidence_class"] == "independent_oracle"


def test_oracle_refuses_to_overwrite(oracle, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["oracle_fixture.py", "--write"])
    with pytest.raises(SystemExit, match="immutable"):
        oracle.main()


def test_calendar_ownership_matches_spec_geometry():
    cal = pd.read_csv(ORACLE_DIR / "calendar_sessions.csv")
    anchors = cal[(cal.calendar == "XNYS") & cal.is_anchor]
    assert len(anchors) == 97
    assert anchors.valuation_owner.value_counts().to_dict() == {"holdout": 12, "outer1": 11, "outer2": 11}
    assert (anchors.role == "warmup").sum() == 20 and (anchors.role == "eligible").sum() == 70
    pos = anchors[anchors.eligible_position > 0].set_index("eligible_position").session_date
    assert (pos[1], pos[41], pos[50], pos[51], pos[60], pos[61], pos[70]) == (
        "2024-01-31", "2024-04-03", "2024-04-17", "2024-04-18", "2024-05-02", "2024-05-03", "2024-05-20")


def test_event_queue_is_sorted_by_spec_6_4_keys():
    q = pd.read_csv(ORACLE_DIR / "event_queue.csv", keep_default_na=False)
    rank = {"XKRX": 0, "XNYS": 1, "": 2}
    keys = list(zip(q.ts, q.priority, q.exchange.map(rank), q.asset_id, q.order_id, strict=True))
    assert keys == sorted(keys)
    assert q.event_id.is_unique and list(q.event_id) == [f"E{i:05d}" for i in range(1, len(q) + 1)]
    pri = {"corporate_action": 0, "cash_accrual": 1, "sell_fill": 2, "fx_conversion": 3, "buy_fill": 4, "valuation": 5}
    assert (q.kind.map(pri) == q.priority).all()


def test_same_timestamp_sells_precede_buys_and_corporate_actions_precede_both():
    q = pd.read_csv(ORACLE_DIR / "event_queue.csv", keep_default_na=False)
    tie = q[q.ts == "2024-02-08T21:00:00Z"].kind.tolist()  # NYSE_C ex-dividend close with fill slots
    assert tie == ["corporate_action", "sell_fill", "buy_fill"]


def test_next_close_is_strictly_after_cutoff_and_matches_spec_16_9(oracle):
    cutoff = pd.Timestamp("2024-01-02T21:15:00Z")
    cals = {n: oracle.xc.get_calendar(n) for n in ("XKRX", "XNYS")}
    assert oracle.z(oracle.next_close(cals, "XKRX", cutoff)[1]) == "2024-01-03T06:30:00Z"
    assert oracle.z(oracle.next_close(cals, "XNYS", cutoff)[1]) == "2024-01-03T21:00:00Z"
    q = pd.read_csv(ORACLE_DIR / "event_queue.csv", keep_default_na=False)
    slots = q[q.kind.isin(["buy_fill", "sell_fill"])]
    first_decision_cutoff = pd.Timestamp("2024-01-31T21:15:00Z")  # eligible position 1
    d1 = slots[slots.order_id.str.startswith("D001-")]
    assert (pd.to_datetime(d1.ts, utc=True) > first_decision_cutoff).all()


def test_panel_never_inserts_a_holiday_zero():
    p = pd.read_csv(ORACLE_DIR / "panel_daily_weekly.csv")
    d = p[p.kind == "daily"]
    for a in ("KRX_A", "KRX_B", "NYSE_C"):
        not_updated = d[~d[f"{a}_updated"]]
        assert not_updated[f"{a}_return"].isna().all(), a  # null, not 0.0
    xnys_only = d[~d.KRX_A_updated & d.NYSE_C_updated]
    assert len(xnys_only) > 0 and (~xnys_only.complete_case).all()  # excluded from complete-case rows (16.8)


def test_panel_complete_case_count_and_genuine_unchanged_return():
    p = pd.read_csv(ORACLE_DIR / "panel_daily_weekly.csv")
    d = p[(p.kind == "daily") & (p.session_date >= "2024-01-02")]
    assert d.complete_case.sum() == 90  # spec 14: 90 complete-case rows in the fixture window
    row = d[d.session_date == "2024-02-27"].iloc[0]
    assert row.KRX_A_updated and row.KRX_A_return == pytest.approx(0.0, abs=1e-12)  # unchanged raw, tr flat
    w = p[p.kind == "weekly"]
    assert (w.session_date.map(lambda s: pd.Timestamp(s).weekday()) == 4).all()
    assert w.cutoff_utc.str.endswith("T23:59:59Z").all()


def test_ledger_oracle_accrual_matches_hand_calculation():
    rows = list(csv.DictReader((ORACLE_DIR / "ledger_projection.csv").open(encoding="utf-8")))
    vals = [r for r in rows if r["kind"] == "valuation"]
    assert vals[0]["nav_krw"] == "1000000000.00"
    # 1e9 KRW * 3.50% * 1 day / 365 = 95,890.41 (rate in force: 3.50% published 2024-01-02T00:00Z)
    assert Decimal(vals[1]["nav_krw"]) - Decimal("1000000000.00") == Decimal("95890.41")


def test_ledger_oracle_first_fill_agrees_with_production_kernel():
    rows = list(csv.DictReader((ORACLE_DIR / "ledger_projection.csv").open(encoding="utf-8")))
    krx_a = next(r for r in rows if r["order_id"] == "S-2024-02-01-KRX_A")
    price = Decimal(krx_a["note"].split("px=")[1].split(";")[0])
    rates = CostRates(Decimal(5), Decimal(5), Decimal(0), Decimal(0))
    res = Ledger(cash=Decimal(1000000000)).post_fill(Fill("KRX_A", "KRW", 3000, price, None), rates)
    assert res["notional"] == Decimal("234558000.00") and res["cost"] == Decimal("234558.00")
    assert krx_a["cumulative_costs"] == "234558.00"  # independent oracle and production kernel agree


def test_ledger_oracle_split_and_dividend_effects():
    rows = list(csv.DictReader((ORACLE_DIR / "ledger_projection.csv").open(encoding="utf-8")))
    split = next(r for r in rows if r["note"] == "split_quantity_adjust")
    before = [r for r in rows if r["ts"] < split["ts"]][-1]
    assert int(split["qty_KRX_B"]) == 2 * int(before["qty_KRX_B"])
    div = next(r for r in rows if r["note"].startswith("dividend_credit"))
    assert Decimal(div["cash_krw"]) > Decimal([r for r in rows if r["ts"] < div["ts"]][-1]["cash_krw"])


def test_ledger_oracle_is_self_financing_between_valuations():
    rows = list(csv.DictReader((ORACLE_DIR / "ledger_projection.csv").open(encoding="utf-8")))
    assert all(Decimal(r["cash_krw"]) >= 0 for r in rows)
    costs = [Decimal(r["cumulative_costs"]) for r in rows]
    assert costs == sorted(costs)  # costs never decrease


def test_hmm_oracle_exact_rationals_match_spec_16_6():
    h = json.loads((ORACLE_DIR / "hmm_filter.json").read_text(encoding="utf-8"))["rational_case"]
    assert h["posteriors"] == [["3/5", "2/5"], ["13/41", "28/41"], ["297/436", "139/436"]]
    assert h["boundary"] == ["297/436", "139/436"] and h["first_oos_predicted"] == ["503/872", "369/872"]


def test_hmm_oracle_agrees_with_production_filter():
    h = json.loads((ORACLE_DIR / "hmm_filter.json").read_text(encoding="utf-8"))
    g = h["gaussian_case"]
    means, var = np.array(g["means"]), np.array(g["variances"])
    obs = np.array(g["observations"])
    lik = np.array([[np.exp(-0.5 * np.sum((o - means[s]) ** 2 / var[s] + np.log(2 * np.pi * var[s])))
                     for s in range(2)] for o in obs])
    got = filter_window(np.array(g["start"]), np.array(g["transition"]), lik)
    np.testing.assert_allclose(got, g["filtered"], atol=1e-12, rtol=1e-10)


def test_sobol_first_eight_are_distinct_and_cover_the_space():
    meta = json.loads((ORACLE_DIR / "sobol_meta.json").read_text(encoding="utf-8"))
    pts = pd.read_csv(ORACLE_DIR / "sobol_points.csv")
    assert meta["cardinality"] == 8 and meta["sizes"] == [2, 2, 2] and meta["seed"] == 1729
    assert meta["dimension_order"] == sorted(meta["dimension_order"])
    assert len(pts) >= 8 and pts.counts_toward_cap.sum() == 8 and pts.trial_hash.nunique() >= 8
    idx = [c for c in pts.columns if c.startswith("idx_")]
    assert len(set(map(tuple, pts[pts.counts_toward_cap][idx].values))) == 8  # all 2*2*2 configurations
    u = pts[[c for c in pts.columns if c.startswith("u_")]].astype(float)
    assert ((u >= 0) & (u < 1)).all().all()


def test_bootstrap_hash_and_regeneration_shape(oracle):
    m = oracle.bootstrap_matrix()
    assert m.shape == (200, 22) and m.min() >= 0 and m.max() <= 21 and m.dtype == np.dtype("<i8")
    stored = (ORACLE_DIR / "bootstrap_indices.sha256").read_text(encoding="utf-8").strip()
    assert stored == _sha(m.tobytes(order="C"))
    # stationary blocks: most steps continue the previous index (p = 1 - 1/5)
    cont = ((m[:, 1:] - m[:, :-1]) % 22 == 1).mean()
    assert 0.75 < cont < 0.9
