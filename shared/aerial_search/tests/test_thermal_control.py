"""Arm A is the existing cache: the new code path must reproduce it (#68).

Needs the real weights and the #66 feature cache on this machine; skipped
otherwise. The cache is only read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerial_search.experiments import feature_cache as fc

ROOT = Path(__file__).resolve().parents[3]


def _main_checkout() -> Path:
    # a worktree's data/ is a symlink to the main checkout's copy
    return (ROOT / "data").resolve().parent


CACHE = _main_checkout() / "outputs/features/siglip2-base-naflex-1024tok"
DATA = ROOT / "data/raw/wisard-full"

pytestmark = pytest.mark.skipif(
    not ((CACHE / "index.jsonl").is_file() and DATA.is_dir()),
    reason="the #66 feature cache or the WiSARD data is not present",
)


def test_the_replicate_path_reproduces_cached_thermal_features():
    index = fc.read_index(CACHE)
    thermal = [e for e in index.values() if e["camera"] == "thermal"][:3]
    frames = [
        fc.Frame(e["path"], e["camera"], e["collection_id"], e["sha256"])
        for e in thermal
    ]
    assert len(frames) == 3
    # the CLI's own loader, with the replicate arm's transform (None)
    extractor = fc._load_naflex("siglip2-base-naflex")(
        1024, 2, fc.input_transform("replicate", "thermal")
    )

    for f in frames:
        stored = np.asarray(fc.read_features(CACHE, f.path))
        fresh = extractor.extract(extractor.prepare(fc._open(DATA / f.path))).array
        # measured 2026-10-03 on the M4 (MPS): identical, not merely close.
        # Same weights, same preprocessing, same device, fp16 throughout. On
        # another device bits can differ, so fall back to the cache's own limit.
        if not np.array_equal(stored, fresh):
            a, b = stored.astype(np.float32), fresh.astype(np.float32)
            drift = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a**2))
            assert drift < fc.VERIFY_TOLERANCE, f.path
