"""RFC 8785 conformance evidence (spec 2.2). Vectors: RFC 8785 text and Appendix B number table."""
from __future__ import annotations

import math
import struct

import pytest

from quant_pipeline.hashing import (
    canonical_text_bytes,
    hash_json_bytes,
    hash_manifest,
    jcs_bytes,
    parse_json_strict,
)


def test_key_ordering_is_utf16_code_unit_order():
    # RFC 8785 section 3.2.3 style sorting: U+20AC (EUR) < U+D83D U+DE00 (emoji, surrogate pair) < U+FB33
    obj = {"\ufb33": 3, "\U0001f600": 2, "\u20ac": 1, "\r": 0, "1": 9}
    parsed = parse_json_strict(jcs_bytes(obj))
    assert list(parsed) == ["\r", "1", "\u20ac", "\U0001f600", "\ufb33"]


def test_whitespace_and_escaping():
    assert jcs_bytes({"b": [1, 2], "a": "x\ny\"\\"}) == b'{"a":"x\\ny\\"\\\\","b":[1,2]}'
    assert jcs_bytes("\u0007") == b'"\\u0007"'
    assert jcs_bytes("\u007f") == "\"\u007f\"".encode()  # DEL is not escaped


# (IEEE-754 bit pattern, expected ECMAScript number string) from RFC 8785 Appendix B
APPENDIX_B = [
    (0x0000000000000000, "0"),
    (0x8000000000000000, "0"),  # negative zero serializes as 0
    (0x0000000000000001, "5e-324"),
    (0x7FEFFFFFFFFFFFFF, "1.7976931348623157e+308"),
    (0x4340000000000000, "9007199254740992"),
    (0xC340000000000000, "-9007199254740992"),
    (0x4430000000000000, "295147905179352830000"),
    (0x44B52D02C7E14AF5, "9.999999999999997e+22"),
    (0x44B52D02C7E14AF6, "1e+23"),
    (0x44B52D02C7E14AF7, "1.0000000000000001e+23"),
    (0x444B1AE4D6E2EF50, "1e+21"),
    (0x3EB0C6F7A0B5ED8D, "0.000001"),
    (0x3EB0C6F7A0B5ED8C, "9.999999999999997e-7"),
]


@pytest.mark.parametrize("bits,expected", APPENDIX_B)
def test_appendix_b_numbers(bits, expected):
    value = struct.unpack(">d", struct.pack(">Q", bits))[0]
    assert jcs_bytes(value).decode() == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected(bad):
    with pytest.raises(ValueError):
        jcs_bytes({"x": bad})


def test_duplicate_keys_and_nan_constants_rejected_on_parse():
    with pytest.raises(ValueError, match="duplicate"):
        parse_json_strict(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite"):
        parse_json_strict(b'{"a":NaN}')


def test_json_hash_ignores_formatting_and_key_order():
    assert hash_json_bytes(b'{ "b": 1,\n "a": [1.0, 2] }') == hash_json_bytes(b'{"a":[1,2],"b":1}')


def test_text_canonicalization():
    assert canonical_text_bytes(b"\xef\xbb\xbfa  \r\nb\t\rc\n\n\n") == b"a\nb\nc\n"
    with pytest.raises(UnicodeDecodeError):
        canonical_text_bytes(b"\xff\xfe")


def test_manifest_hash_is_order_independent_and_relative_posix_only():
    a = hash_manifest({"b/x": "sha256:1", "a": "sha256:2"})
    assert a == hash_manifest({"a": "sha256:2", "b/x": "sha256:1"})
    with pytest.raises(ValueError):
        hash_manifest({"C:\\x": "sha256:1"})
    assert a.startswith("sha256:")
