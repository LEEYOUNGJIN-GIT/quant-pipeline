"""Canonical content hashing (spec 3.2). All hashes are SHA-256 with a ``sha256:`` prefix."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import rfc8785

PREFIX = "sha256:"


def _sha(data: bytes) -> str:
    return PREFIX + hashlib.sha256(data).hexdigest()


def canonical_text_bytes(raw: bytes) -> bytes:
    """Markdown/plain text: strict UTF-8, no BOM, LF endings, no trailing blanks, one final LF."""
    text = raw.decode("utf-8")  # invalid bytes raise UnicodeDecodeError
    text = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    return ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")


def hash_text(raw: bytes) -> str:
    return _sha(canonical_text_bytes(raw))


def reject_non_finite(obj: Any) -> None:
    """Spec 3.1/3.2: non-finite numbers are never hashable."""
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("non-finite number in canonical data")
    if isinstance(obj, dict):
        for v in obj.values():
            reject_non_finite(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            reject_non_finite(v)


def jcs_bytes(obj: Any) -> bytes:
    """RFC 8785 JCS serialization. Non-finite values and non-string keys are rejected."""
    reject_non_finite(obj)
    return rfc8785.dumps(obj)


def hash_json_obj(obj: Any) -> str:
    return _sha(jcs_bytes(obj))


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate JSON key: {k!r}")
        d[k] = v
    return d


def _bad_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {name}")


def parse_json_strict(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes, parse_constant=_bad_constant)


def hash_json_bytes(raw: bytes) -> str:
    return hash_json_obj(parse_json_strict(raw))


def hash_file_raw(path: Path) -> str:
    return _sha(Path(path).read_bytes())


def hash_manifest(entries: dict[str, str]) -> str:
    """Directory/manifest hash: POSIX relative paths sorted by code point paired with content hashes."""
    for p in entries:
        if "\\" in p or p.startswith("/"):
            raise ValueError(f"manifest path must be relative POSIX: {p!r}")
    ordered = {p: entries[p] for p in sorted(entries)}
    return hash_json_obj(ordered)


def sha256_bytes(data: bytes) -> str:
    """Raw-byte SHA-256 with the ``sha256:`` prefix (artifact integrity; not a canonicalized content hash)."""
    return _sha(data)
