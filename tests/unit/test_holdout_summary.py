"""The opened-run summary prints aggregates only and refuses a directory that was not opened (tools/holdout_summary.py)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("holdout_summary", ROOT / "tools" / "holdout_summary.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

REPORT = """# Report
## 6. Other
text
## 7. Holdout status
The final holdout was opened once.
| name | label | total return | sharpe | max drawdown |
| strategy | strategy | 0.1000 | 0.50 | -0.0800 |
## 8. Limitations
more
"""


def make(tmp_path, audit=True, report=True):
    run = tmp_path / "real3"
    cp = run / "checkpoints" / "fold_id=holdout" / "x"
    cp.mkdir(parents=True)
    if audit:
        (cp / "holdout_audit.json").write_text(json.dumps({"opening_number": 1, "max_openings_per_lineage": 1, "pre_holdout_unchanged": True,
                                                           "audit_hash": "sha256:" + "a" * 64, "refit_policy": "x", "minima": {"met": True}}), encoding="utf-8")
    if report:
        post = run / "checkpoints" / "fold_id=global" / "trial_hash=post-holdout"
        post.mkdir(parents=True)
        (post / "report.md").write_text(REPORT, encoding="utf-8")
    return run


def test_the_holdout_section_is_cut_between_its_own_headings():
    lines = H.holdout_section(REPORT)
    assert lines[0].startswith("## 7. Holdout status") and all(not x.startswith("## 8") for x in lines) and any("strategy" in x for x in lines)


def test_an_opened_run_prints_audit_facts_and_the_section(tmp_path, capsys):
    assert H.main(["x", str(make(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert '"pre_holdout_unchanged": true' in out and "Holdout status" in out and "0.1000" in out


def test_a_run_that_was_not_opened_is_refused(tmp_path, capsys):
    assert H.main(["x", str(make(tmp_path, audit=False))]) == 1
    assert "not opened" in capsys.readouterr().out


def test_an_opened_run_without_the_report_is_refused(tmp_path, capsys):
    assert H.main(["x", str(make(tmp_path, report=False))]) == 1
