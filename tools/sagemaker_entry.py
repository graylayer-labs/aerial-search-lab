"""SageMaker training job entry point (#113): runs `aerial_search.cloud.entry`.

The container's toolkit unpacks the pinned source archive into /opt/ml/code and
runs this file. It only puts `src/` on the path; the design is in
`src/aerial_search/cloud/entry.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_search.cloud import entry  # noqa: E402

if __name__ == "__main__":
    entry.main()
