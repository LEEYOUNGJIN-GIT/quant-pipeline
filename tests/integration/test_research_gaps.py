"""A provider gap (a session on which an asset has no observation) is excluded from the complete-case rows, never filled (INV-DATA-001, spec 5.1 rules 5-7).

The first live collection found four XKRX sessions on which Yahoo had no bar for any Korean instrument. The reduced SYNTHETIC research run below has the
same kind of gaps (one asset of each market); the engine must complete, the report must name the excluded sessions, and the artifacts must verify.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))


def _mutate(tables):
    tr = tables["prices_total_return"]
    gaps = {"R01": set(tr.query("instrument_id == 'R01'").exchange_session.iloc[[150, 151, 300]]),
            "R02": set(tr.query("instrument_id == 'R02'").exchange_session.iloc[[220]])}
    for name in ("prices_raw", "prices_total_return"):
        t = tables[name]
        keep = ~t.apply(lambda r: r.exchange_session in gaps.get(r.instrument_id, ()), axis=1)
        tables[name] = t[keep]


def test_the_engine_completes_on_a_snapshot_with_provider_gaps_and_the_report_names_them(tmp_path):
    from quant_pipeline.artifact_check import check_run
    from quant_pipeline.research_pipeline import ResearchPipeline
    from support.research_smoke import SMOKE_WINDOW, write_smoke_config
    from support.synthetic_research import build_research_repo

    root, _ = build_research_repo(tmp_path, start=SMOKE_WINDOW[0], end=SMOKE_WINDOW[1], mutate=_mutate)
    cfg = write_smoke_config(root)
    p = ResearchPipeline(root, root / "out", "gap", config=cfg, allow_synthetic=True)
    p.run()
    excluded = sorted(p._unobserved)
    assert 3 <= len(excluded) <= 4  # R01's two adjacent sessions and one more, R02's one (a shared date would merge them)
    assert not set(excluded) & {s.strftime("%Y-%m-%d") for s in p.rm.complete_case}
    reports = list((root / "out" / "gap").rglob("stage=12_bootstrap_and_reports/**/report.md"))
    assert len(reports) == 1
    report = reports[0].read_text(encoding="utf-8")
    assert "never filled" in report and all(d in report for d in excluded)
    result = check_run(root / "out" / "gap")
    assert result["passed"], [c for c in result["checks"] if not c["ok"]]
