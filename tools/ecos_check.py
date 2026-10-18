"""Confirm that the configured ECOS statistic/item is the overnight call rate before a snapshot is accepted. Prints names only, never the key."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.collector import ProviderError
from quant_pipeline.ecos_adapter import DEFAULT_ITEM_CODE, DEFAULT_STAT_CODE, EXPECTED_ITEM_WORDS, RealEcosClient


def main() -> int:
    try:
        names = RealEcosClient().item_names(DEFAULT_STAT_CODE)
    except ProviderError as e:
        print(f"ECOS item list failed: {e}")
        return 2
    name = names.get(DEFAULT_ITEM_CODE)
    print(f"statistic {DEFAULT_STAT_CODE} has {len(names)} items; configured item {DEFAULT_ITEM_CODE} is: {name!r}")
    for code, label in sorted(names.items()):
        print(f"  {code}: {label}")
    if name is None or not any(w in name.lower() for w in EXPECTED_ITEM_WORDS):
        print("FAIL: the configured item is not the overnight call rate; fix ecos_adapter.DEFAULT_ITEM_CODE from the list above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
