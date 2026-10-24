"""The GPU job request, byte for byte against a committed copy (#191).

The CPU work (#188, #191) shares the launcher with the GPU `train-baseline`
job. This builds the GPU request from fixed inputs and compares it with
`fixtures/gpu_request.json`, so any change to it shows. If a change is meant,
regenerate the fixture with `UPDATE_GOLDEN=1 uv run pytest <this file>` and
say why in the commit.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from aerial_search.cloud import launcher

FIXTURE = Path(__file__).parent / "fixtures" / "gpu_request.json"


def gpu_request() -> dict:
    return launcher.build_request(
        logical_run=(
            "detection-baseline/fasterrcnn_resnet50_fpn-tiled/"
            "rgb-220109_Baker-10pct-seed7"
        ),
        attempt=1,
        commit="a" * 40,
        source_sha256="b" * 64,
        manifests_sha256="c" * 64,
        constraints_sha256="d" * 64,
        baseline_args=["rgb", "--percent", "10", "--fold", "220109_Baker"],
        role_arn="arn:aws:iam::123456789012:role/r",
        spot=True,
        max_runtime_hours=12.0,
        max_wait_hours=18.0,
        scratch=False,
        now=datetime(2026, 10, 24, 12, tzinfo=UTC),
    )


def test_the_gpu_request_is_unchanged() -> None:
    text = json.dumps(gpu_request(), indent=2) + "\n"
    if os.environ.get("UPDATE_GOLDEN"):
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(text)
    assert text == FIXTURE.read_text()
