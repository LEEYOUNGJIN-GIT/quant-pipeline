"""Capability probe and provenance audit (spec 18, 19 Phase 7, check ``INV-GOV-001``).

Every field is either observed or an explicit null with a reason. Nothing is inferred from file names, and a local run is never
described as CI: CI facts are read only from the environment variables a real GitHub Actions job sets.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CI_VARS = ("GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_WORKFLOW", "GITHUB_REPOSITORY", "GITHUB_REF")


def _git(root: Path, *args: str) -> tuple[int, str]:
    exe = shutil.which("git")
    if exe is None:
        return 127, "git executable not found"
    r = subprocess.run([exe, *args], cwd=root, capture_output=True, text=True, check=False, timeout=30)
    return r.returncode, (r.stdout or r.stderr).strip()


def _null(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def probe_git(root: Path) -> dict[str, Any]:
    rc, inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if rc != 0 or inside != "true":
        return {"is_repository": False, "head_sha": _null("not a git work tree"), "branch": _null("not a git work tree"),
                "dirty": _null("not a git work tree"), "remote_origin": _null("not a git work tree")}
    rc, sha = _git(root, "rev-parse", "HEAD")
    head = {"value": sha, "reason": None} if rc == 0 and SHA_RE.match(sha) else _null("repository has no commit yet")
    rc, branch = _git(root, "symbolic-ref", "--short", "HEAD")
    rc2, status = _git(root, "status", "--porcelain")
    rc3, remote = _git(root, "remote", "get-url", "origin")
    return {"is_repository": True, "head_sha": head,
            "branch": {"value": branch, "reason": None} if rc == 0 else _null("detached or unborn HEAD"),
            "dirty": {"value": bool(status), "reason": None} if rc2 == 0 else _null("git status failed"),
            "remote_origin": {"value": remote, "reason": None} if rc3 == 0 and remote else _null("no remote named origin")}


def probe_github_cli() -> dict[str, Any]:
    exe = shutil.which("gh")
    if exe is None:
        return {"gh_installed": False, "authenticated": _null("gh executable not found")}
    r = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, check=False, timeout=30)
    # only the exit status is kept: the output can name accounts and must not be copied into evidence
    return {"gh_installed": True, "authenticated": {"value": r.returncode == 0, "reason": None}}


def probe_ci(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    e = os.environ if env is None else env
    in_ci = e.get("GITHUB_ACTIONS") == "true"
    if not in_ci:
        return {"in_ci": False, "reason": "GITHUB_ACTIONS is not 'true': this is a local run and is not described as CI",
                "sha": None, "run_id": None, "workflow": None, "repository": None}
    sha = e.get("GITHUB_SHA", "")
    return {"in_ci": True, "reason": None, "sha": sha if SHA_RE.match(sha) else None, "run_id": e.get("GITHUB_RUN_ID") or None,
            "workflow": e.get("GITHUB_WORKFLOW") or None, "repository": e.get("GITHUB_REPOSITORY") or None}


def sha_audit(git: dict[str, Any], ci: dict[str, Any]) -> dict[str, Any]:
    """Verified or null, never invented. In CI the checked-out HEAD must equal the SHA the runner reports."""
    head = git["head_sha"]["value"]
    if not ci["in_ci"]:
        return {"status": "local_only", "head_sha": head, "ci_sha": None,
                "note": "no CI evidence exists; any CI claim would be unsupported"}
    if head is None or ci["sha"] is None:
        return {"status": "unverifiable", "head_sha": head, "ci_sha": ci["sha"], "note": "a SHA is missing"}
    return {"status": "verified" if head == ci["sha"] else "mismatch", "head_sha": head, "ci_sha": ci["sha"],
            "note": "HEAD equals the runner-reported SHA" if head == ci["sha"] else "checked-out HEAD differs from GITHUB_SHA"}


def probe(root: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    git, ci = probe_git(root), probe_ci(env)
    return {"schema": "capability-probe-1", "git": git, "github_cli": probe_github_cli(), "ci": ci, "sha_audit": sha_audit(git, ci),
            "blocking": False, "note": "optional GitHub capability never blocks a fixture run"}
