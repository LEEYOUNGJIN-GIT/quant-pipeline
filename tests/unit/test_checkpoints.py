"""Checkpoint store (spec 12.2, 12.3): keys, atomic write, validation, immutability and dependency-chain resume."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_pipeline.checkpoints import (
    MARKER,
    STAGES,
    CheckpointExists,
    CheckpointKey,
    CheckpointStore,
    LineageMismatch,
)

LIN = {"spec": "s1", "config": "c1", "data": "d1", "lock": "l1", "code": "k1"}


def key(stage="10_event_time_execution_and_accounting", fold="f1", trial="t1", refit="none", run="run"):
    return CheckpointKey(run, fold, trial, stage, refit)


def write(store, k=None, data=None, lineage=None, parents=None):
    return store.write(k or key(), data or {"a.json": b'{"x": 1}', "b.csv": b"1,2\n"}, lineage=lineage or LIN, parents=parents,
                       meta={"rows": 2})


def test_the_path_encodes_the_full_key_not_the_stage_alone(tmp_path):
    s = CheckpointStore(tmp_path)
    p = s.path(key(fold="outer-1", trial="selected-abc", refit="r41"))
    plain = Path(str(p).removeprefix("\\\\?\\"))  # drop the Windows extended-length prefix if one was added
    assert plain.relative_to(tmp_path.resolve()).parts == ("run", "checkpoints", "fold_id=outer-1", "trial_hash=selected-abc",
                                             "stage=10_event_time_execution_and_accounting", "refit_id=r41")
    assert s.path(key(fold="a")) != s.path(key(fold="b")) and s.path(key(trial="a")) != s.path(key(trial="b"))
    assert s.path(key(refit="r1")) != s.path(key(refit="r6"))


def test_invalid_key_components_are_rejected():
    for bad in ({"stage": "nonsense"}, {"fold": "a/b"}, {"trial": ""}, {"refit": "x\\y"}):
        with pytest.raises(ValueError):
            key(**bad)
    assert len(STAGES) == 14 and STAGES[0] == "00_capability_probe" and STAGES[-1] == "13_verification"


def test_a_written_checkpoint_validates_and_round_trips(tmp_path):
    s = CheckpointStore(tmp_path)
    write(s)
    assert s.validate(key(), LIN) == (True, "ok")
    assert s.read(key(), "a.json", LIN) == b'{"x": 1}' and s.meta(key()) == {"rows": 2}
    body = json.loads((s.path(key()) / "checkpoint.json").read_text(encoding="utf-8"))
    assert body["lineage"] == LIN and set(body["artifacts"]) == {"a.json", "b.csv"} and (s.path(key()) / MARKER).exists()


def test_prior_artifacts_are_never_overwritten(tmp_path):
    s = CheckpointStore(tmp_path)
    write(s)
    with pytest.raises(CheckpointExists):
        write(s, data={"a.json": b"different"})
    assert s.read(key(), "a.json", LIN) == b'{"x": 1}'


def test_a_failed_write_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    s = CheckpointStore(tmp_path)
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        write(s)
    monkeypatch.undo()
    assert not s.path(key()).exists()
    assert not [p for p in s.path(key()).parent.glob(".*tmp-*")]  # the temporary sibling is removed too
    write(s)  # and the key is still writable afterwards
    assert s.validate(key(), LIN)[0]


@pytest.mark.parametrize("damage,reason", [
    (lambda d: (d / MARKER).unlink(), "no_completion_marker"),
    (lambda d: (d / "a.json").write_bytes(b"tampered"), "artifact_hash_mismatch:a.json"),
    (lambda d: (d / "b.csv").unlink(), "missing_artifact:b.csv"),
    (lambda d: (d / "checkpoint.json").write_text("{not json", encoding="utf-8"), "unreadable_checkpoint_json"),
    (lambda d: (d / MARKER).write_text("wrong", encoding="utf-8"), "content_hash_mismatch"),
])
def test_missing_marker_partial_tampered_or_corrupt_checkpoints_are_invalid(tmp_path, damage, reason):
    s = CheckpointStore(tmp_path)
    write(s)
    damage(s.path(key()))
    ok, why = s.validate(key(), LIN)
    assert not ok and why == reason
    with pytest.raises(ValueError):
        s.read(key(), "a.json", LIN)


def test_schema_mismatch_is_invalid(tmp_path):
    s = CheckpointStore(tmp_path)
    write(s)
    f = s.path(key()) / "checkpoint.json"
    body = json.loads(f.read_text(encoding="utf-8"))
    body["schema"] = "checkpoint-0"
    f.write_text(json.dumps(body), encoding="utf-8")
    assert s.validate(key(), LIN) == (False, "schema_mismatch")


def test_a_lineage_mismatch_is_refused_not_silently_recomputed(tmp_path):
    s = CheckpointStore(tmp_path)
    write(s)
    changed = {**LIN, "spec": "s2", "code": "k2"}
    ok, why = s.validate(key(), changed)
    assert not ok and why == "lineage_mismatch:code,spec"
    with pytest.raises(LineageMismatch):
        s.read(key(), "a.json", changed)
    with pytest.raises(LineageMismatch):
        s.latest_valid([key()], changed)
    with pytest.raises(LineageMismatch):
        s.discard_invalid(key(), changed)
    assert s.validate(key(), LIN)[0]  # nothing was touched


def test_resume_picks_the_last_valid_prefix_of_the_dependency_chain(tmp_path):
    s = CheckpointStore(tmp_path)
    chain = [key(stage="03_canonical_panel", fold="global"), key(stage="04_features_and_pca", refit="r1"),
             key(stage="05_covariance_and_asynchrony", refit="r1"), key(stage="11_oos_evaluation", fold="outer-1")]
    for i, k in enumerate(chain[:3]):
        write(s, k, parents=chain[:i])
    assert s.latest_valid(chain, LIN) == 3
    (s.path(chain[1]) / MARKER).unlink()  # a gap invalidates everything after it, even if later entries look fine
    assert s.latest_valid(chain, LIN) == 1


def test_discard_invalid_quarantines_a_partial_checkpoint_and_keeps_valid_ones(tmp_path):
    s = CheckpointStore(tmp_path)
    write(s)
    assert s.discard_invalid(key(), LIN) is None and s.validate(key(), LIN)[0]
    (s.path(key()) / MARKER).unlink()
    assert s.discard_invalid(key(), LIN) == "no_completion_marker" and not s.path(key()).exists()
    write(s)  # the key can now be recomputed, and the damaged copy is kept aside for audit
    assert any(p.name.startswith(".invalid-") for p in s.path(key()).parent.iterdir())


def test_artifact_names_cannot_escape_or_shadow_reserved_files(tmp_path):
    s = CheckpointStore(tmp_path)
    for bad in ("../x", "a/b", MARKER, "checkpoint.json"):
        with pytest.raises(ValueError):
            s.write(key(trial=f"t-{abs(hash(bad))}"), {bad: b"x"}, lineage=LIN)


def test_parent_keys_are_recorded_for_the_graph(tmp_path):
    s = CheckpointStore(tmp_path)
    parent = key(stage="04_features_and_pca", refit="r1")
    write(s, parent)
    write(s, key(stage="05_covariance_and_asynchrony", refit="r1"), parents=[parent])
    body = json.loads((s.path(key(stage="05_covariance_and_asynchrony", refit="r1")) / "checkpoint.json").read_text(encoding="utf-8"))
    assert body["parents"][0]["stage"] == "04_features_and_pca" and Path(tmp_path).exists()


def test_paths_longer_than_the_windows_limit_still_work(tmp_path):
    deep = tmp_path / ("d" * 70) / ("e" * 70) / ("f" * 70)
    s = CheckpointStore(deep)
    k = key(stage="11_oos_evaluation", fold="outer-1-x0", trial="selected-abcdef123456")
    s.write(k, {"strategy.ledger_rows.csv": b"x\n"}, lineage=LIN)
    assert len(str(s.path(k))) > 260 and s.validate(k, LIN) == (True, "ok")
    assert s.read(k, "strategy.ledger_rows.csv", LIN) == b"x\n"
