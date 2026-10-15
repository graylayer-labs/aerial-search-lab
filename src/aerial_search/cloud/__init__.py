"""Cloud runs: one `train-baseline` as one SageMaker training job (#113).

`source` packages and hashes the code, `cost` is the pre-launch guard,
`launcher` builds and sends the job, `entry` is what runs inside it.
"""

import re


def safe_name(text: str) -> str:
    """`text` with anything outside letters, digits, dot, underscore and
    hyphen replaced by a hyphen: safe in a run name and an S3 key."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
