"""Golden outputs of whole experiment runs, shared by the projects' tests (#216).

A characterization test runs an experiment on a tiny synthetic input and
compares what it wrote (`run.json`, `training.json`, the test report, the
saved weights) with `tests/fixtures/golden/<name>.json` beside the test.
On the platform and torch version that recorded the fixture the comparison is
exact, to the last digit of every float and the order of every key; elsewhere
floats may differ by rounding (rtol 1e-3) and weight digests are skipped.
`UPDATE_GOLDEN=1 uv run pytest ...` rewrites the fixtures: only when an output
is meant to change, and the diff then says what changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import pytest
import torch

# Fields of run.json that differ on every run or machine.
VOLATILE = ("started_at", "finished_at", "machine", "packages")
EXACT_ONLY = {"sha256_exact", "model_id"}  # digests of floats; this platform only


def _here() -> str:
    return f"{platform.system()}-{platform.machine()}-torch{torch.__version__}"


def weights(state: Any) -> Any:
    """Tensors as dtype, shape, sum and digest; containers walked."""
    if isinstance(state, torch.Tensor):
        tensor = state.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sum": float(tensor.double().sum()) if tensor.numel() else 0.0,
            "sha256_exact": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(state, dict):
        return {str(k): weights(v) for k, v in state.items()}
    if isinstance(state, list | tuple):
        return [weights(v) for v in state]
    return state


def _close(expected: Any, actual: Any, where: str) -> None:
    assert type(expected) is type(actual) or {type(expected), type(actual)} == {
        int,
        float,
    }, f"{where}: {expected!r} != {actual!r}"
    if isinstance(expected, dict):
        assert list(expected) == list(actual), f"{where}: keys differ"
        for key in expected:
            if key not in EXACT_ONLY:
                _close(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), f"{where}: lengths differ"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _close(e, a, f"{where}[{i}]")
    elif isinstance(expected, float) and not math.isnan(expected):
        assert math.isclose(expected, actual, rel_tol=1e-3, abs_tol=1e-6), where
    elif isinstance(expected, float):
        assert math.isnan(actual), where
    else:
        assert expected == actual, f"{where}: {expected!r} != {actual!r}"


class Golden:
    """Reads a run's outputs and checks them against a stored fixture."""

    def __init__(self, folder: Path, tmp_path: Path) -> None:
        self.folder, self.tmp_path = folder, tmp_path

    def read_json(self, path: Path) -> Any:
        """A JSON file with the test's temporary directory written as `<tmp>`."""
        return json.loads(path.read_text().replace(str(self.tmp_path), "<tmp>"))

    def run_record(self, directory: Path) -> dict[str, Any]:
        record = self.read_json(directory / "run.json")
        return {k: v for k, v in record.items() if k not in VOLATILE}

    weights = staticmethod(weights)

    def check(self, name: str, outputs: Any) -> None:
        path = self.folder / f"{name}.json"
        actual = json.loads(json.dumps(outputs))  # tuples as lists, as stored
        if os.environ.get("UPDATE_GOLDEN") == "1":
            path.parent.mkdir(parents=True, exist_ok=True)
            stored = {"platform": _here(), "outputs": actual}
            path.write_text(json.dumps(stored) + "\n")
            return
        stored = json.loads(path.read_text())
        _close(stored["outputs"], actual, name)
        if stored["platform"] == _here():
            assert json.dumps(actual) == json.dumps(stored["outputs"]), name


@pytest.fixture
def golden(request: pytest.FixtureRequest, tmp_path: Path) -> Golden:
    """Checks a run's outputs against `fixtures/golden/` beside the test."""
    return Golden(request.path.parent / "fixtures" / "golden", tmp_path)
