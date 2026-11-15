"""Golden outputs of whole experiment runs, shared by the projects' tests (#216).

A characterization test runs an experiment on a tiny synthetic input and
compares what it wrote (`run.json`, `training.json`, the test report, the
saved weights) with `tests/fixtures/golden/<name>.json` beside the test.
On the platform and torch version that recorded the fixture (the Mac) the
comparison is exact, to the last digit of every float and the order of every
key. Training is not bit-reproducible across CPUs (float rounding, fused
multiply-add), and a few steps amplify the difference, so elsewhere (CI on
Linux) only the structure is compared, with a warning saying so: keys and
their order, list lengths, types, strings and integers such as step numbers.
`GOLDEN_STRUCTURE_ONLY=1` forces that path on the Mac.
`UPDATE_GOLDEN=1 uv run pytest ...` rewrites the fixtures: only when an output
is meant to change, and the diff then says what changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

# Fields of run.json that differ on every run or machine.
VOLATILE = ("started_at", "finished_at", "machine", "packages")
# Off the recording platform: digests of floats, and counts that follow from
# the float values (predictions above the score floor, ties, the chosen step).
FREE = {
    "sha256_exact",
    "model_id",
    "n_predictions",
    "n_predictions_tied",
    "test_predictions_tied",
    "best_step",
}


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


def _floats(value: Any) -> bool:
    """A numeric array: numbers (or None) only, none of them a string or bool,
    and empty or holding a float."""
    leaves: list[Any] = []

    def walk(v: Any) -> bool:
        if isinstance(v, list):
            return all(walk(x) for x in v)
        leaves.append(v)
        return v is None or (isinstance(v, int | float) and not isinstance(v, bool))

    return walk(value) and (not leaves or any(isinstance(v, float) for v in leaves))


def _same_shape(expected: Any, actual: Any, where: str) -> None:
    """Keys and their order, lengths, types, strings and integers equal;
    floats, numeric arrays and the outcome counts in `FREE` only typed."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{where}: {actual!r} is not a dict"
        assert list(expected) == list(actual), f"{where}: keys differ"
        for key in expected:
            if key not in FREE:
                _same_shape(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{where}: {actual!r} is not a list"
        if _floats(expected) and _floats(actual):
            return
        assert len(expected) == len(actual), f"{where}: lengths differ"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _same_shape(e, a, f"{where}[{i}]")
    elif isinstance(expected, int) and not isinstance(expected, bool):
        assert (type(actual) is int and expected == actual) or (
            isinstance(actual, float) and actual.is_integer() and expected == actual
        ), f"{where}: {expected!r} != {actual!r}"
    elif expected is None or isinstance(expected, float):
        assert actual is None or type(actual) in (int, float), f"{where}: {actual!r}"
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
        forced = os.environ.get("GOLDEN_STRUCTURE_ONLY") == "1"
        if stored["platform"] == _here() and not forced:
            assert json.dumps(actual) == json.dumps(stored["outputs"]), name
            return
        warnings.warn(
            f"golden {name}: values are pinned on {stored['platform']}, this is "
            f"{_here()}{' (forced)' if forced else ''}; only the structure is "
            "compared here. Run the golden tests on the Mac to check values.",
            stacklevel=2,
        )
        _same_shape(stored["outputs"], actual, name)


@pytest.fixture
def golden(request: pytest.FixtureRequest, tmp_path: Path) -> Golden:
    """Checks a run's outputs against `fixtures/golden/` beside the test."""
    return Golden(request.path.parent / "fixtures" / "golden", tmp_path)
