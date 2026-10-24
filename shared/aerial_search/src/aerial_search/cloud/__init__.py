"""Cloud runs: one `train-baseline` as one SageMaker training job (#113).

`source` packages and hashes the code, `cost` is the pre-launch guard,
`launcher` builds and sends the job, `entry` is what runs inside it.
"""

import re


def safe_name(text: str) -> str:
    """`text` with anything outside letters, digits, dot, underscore and
    hyphen replaced by a hyphen: safe in a run name and an S3 key."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


_PLAIN = re.compile(r"[A-Za-z0-9._-]+")


def plain_cache_name(cache: str) -> bool:
    """`<parent>/<directory>`: two names of letters, digits, dot, underscore
    and hyphen, neither `.` nor `..`. The launcher refuses any other before
    sending, and the job before linking (#191)."""
    parts = cache.split("/")
    return len(parts) == 2 and all(
        _PLAIN.fullmatch(p) and p not in (".", "..") for p in parts
    )
