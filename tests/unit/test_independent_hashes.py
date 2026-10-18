"""Independent recomputation of canonical hashes (INV-HASH-001): a standalone stdlib-only serializer, sharing no code with
``quant_pipeline.hashing`` and not using the ``rfc8785`` library, must reproduce the production hashes of real project objects."""
from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path

import pytest

from quant_pipeline.config import load_config
from quant_pipeline.hashing import hash_json_obj, hash_manifest, sha256_bytes

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "data" / "fixtures" / "fixture_3asset"


def es_number(x: float) -> str:
    """ECMAScript Number::toString from the shortest round-trip digits (ECMA-262 7.1.12.1), written independently of the library."""
    if x == 0:
        return "0"
    if math.isnan(x) or math.isinf(x):
        raise ValueError("non-finite")
    sign = "-" if x < 0 else ""
    d = Decimal(repr(abs(x))).normalize()
    digits = "".join(map(str, d.as_tuple().digits))
    k, n = len(digits), len(digits) + d.as_tuple().exponent  # value = 0.digits * 10**n
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant = digits[0] + ("." + digits[1:] if k > 1 else "")
        out = f"{mant}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + out


def jcs(v) -> str:
    """RFC 8785 for the value domain used by this project's hashed objects: null, bool, int, plain-decimal float, str, list, dict."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return es_number(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(jcs(x) for x in v) + "]"
    if isinstance(v, dict):
        keys = sorted(v, key=lambda k: k.encode("utf-16-be"))  # UTF-16 code unit order
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + jcs(v[k]) for k in keys) + "}"
    raise TypeError(type(v))


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_independent_serializer_reproduces_the_production_hash_of_real_objects():
    cfg = load_config(ROOT / "configs" / "fixture_3asset.yaml").model_dump(mode="json")
    objects = {"manifest": json.loads((FIX / "manifest.json").read_text(encoding="utf-8")),
               "lock": json.loads((FIX / "fixture_generation.lock.json").read_text(encoding="utf-8")), "config": cfg}
    for name, obj in objects.items():
        assert sha(jcs(obj)) == hash_json_obj(obj), name


def test_the_manifest_hashes_match_independently_recomputed_raw_file_hashes():
    manifest = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        raw = "sha256:" + hashlib.sha256((FIX / name).read_bytes()).hexdigest()
        assert raw == entry["raw_sha256"], name
        assert sha256_bytes((FIX / name).read_bytes()) == raw


def test_the_directory_manifest_hash_is_the_hash_of_sorted_path_hash_pairs():
    entries = {"b/x.csv": "sha256:" + "1" * 64, "a.csv": "sha256:" + "2" * 64, "B.csv": "sha256:" + "3" * 64}
    expected = sha('{{"B.csv":"{}","a.csv":"{}","b/x.csv":"{}"}}'.format(entries["B.csv"], entries["a.csv"], entries["b/x.csv"]))
    assert hash_manifest(entries) == expected  # uppercase sorts before lowercase: code point order, not locale order


def test_key_order_utf16_ordering_and_escapes_agree_with_production():
    tricky = {"b": 1, "a": [True, None, "é中", "line\nbreak\t\"q\""], "\U0001f600": 2.5, "～": 3, "z": -0.0 + 10}
    assert sha(jcs(tricky)) == hash_json_obj(tricky)  # includes a non-BMP key that sorts before a BMP key in UTF-16 order


def test_number_formatting_matches_the_ecmascript_rules_and_production():
    cases = {1e-9: "1e-9", 1e-7: "1e-7", 1.5e-6: "0.0000015", 0.000001: "0.000001", 123456789.125: "123456789.125", 1e21: "1e+21",
             1e20: "100000000000000000000", 0.1: "0.1", -2.5e-8: "-2.5e-8", 5.0: "5", 1.7976931348623157e308: "1.7976931348623157e+308"}
    for x, text in cases.items():
        assert es_number(x) == text
        assert sha(jcs({"x": x})) == hash_json_obj({"x": x})
    with pytest.raises(ValueError):
        jcs(float("nan"))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", sha("x"))
