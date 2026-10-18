"""Shared expensive evidence: the reduced SYNTHETIC research run is built once per session (about four minutes) and reused by the modules that
inspect it (test_research_pipeline) or copy it (test_research_holdout, which opens the holdout on its own copy and never touches the original)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def research_base(tmp_path_factory):
    from quant_pipeline.research_pipeline import ResearchPipeline
    from support.research_smoke import SMOKE_WINDOW, write_smoke_config
    from support.synthetic_research import build_research_repo

    root, _ = build_research_repo(tmp_path_factory.mktemp("research_base"), start=SMOKE_WINDOW[0], end=SMOKE_WINDOW[1])
    cfg = write_smoke_config(root)
    out = root / "out"
    p = ResearchPipeline(root, out, "r1", config=cfg, allow_synthetic=True)
    proj = p.run()
    pristine = root.parent / "pristine"  # taken before any test touches the original
    shutil.copytree(root, pristine, symlinks=True)
    return {"root": root, "cfg": cfg, "out": out, "p": p, "proj": proj, "pristine": pristine}
