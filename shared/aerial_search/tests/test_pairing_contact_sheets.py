import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "pairing_contact_sheets.py"


def _tool():
    spec = importlib.util.spec_from_file_location("pairing_contact_sheets", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sheets_come_only_from_a_manifest() -> None:
    # An equal-frame-number pairing of two directories ignores the collection's
    # offset and clip numbers, so it could show a pairing the code would not make.
    parser = _tool().build_parser()

    args = parser.parse_args(["raw", "all_pairs.jsonl", "out"])
    assert (args.source, args.manifest, args.output) == (
        Path("raw"),
        Path("all_pairs.jsonl"),
        Path("out"),
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["raw", "--pair", "a_VIS_1", "a_IR_1", "out"])
    with pytest.raises(SystemExit):
        parser.parse_args(["raw", "out"])
