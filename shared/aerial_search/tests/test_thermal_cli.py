"""The command-line side of the thermal input ablation (#68)."""

import json

import pytest

from aerial_search import cli


def write_arm_run(root, arm, ap25) -> None:
    run = root / "thermal-A-10"
    run.mkdir(parents=True)
    config = {"camera": "thermal", "fold": "A", "percent": 10, "input_handling": arm}
    record = {"status": "completed", "scratch": False, "config": config, "seed": 7}
    (run / "run.json").write_text(json.dumps(record))
    overall = {"ap_iou25": ap25, "ap_iou50": 0.1, "recall_at_fppi": {}}
    metrics = {"overall": overall, "by_size": {}}
    (run / "detection_metrics.json").write_text(json.dumps(metrics))


def test_head_table_ablation_puts_each_arm_directory_in_one_table(
    tmp_path, capsys
) -> None:
    write_arm_run(tmp_path / "a", "replicate", 0.2)
    write_arm_run(tmp_path / "b", "equalise", 0.3)
    flags = ["--camera", "thermal", "--percent", "10"]

    cli.main(
        ["head-table", "--ablation", str(tmp_path / "a"), str(tmp_path / "b"), *flags]
    )

    out = capsys.readouterr().out
    assert "| replicate | 0.200 / 0.100" in out
    assert "| equalise | 0.300 / 0.100" in out
    assert "+0.100" in out  # the change against replicate
    with pytest.raises(SystemExit, match="--ablation"):
        cli.main(["head-table", str(tmp_path / "a"), str(tmp_path / "b")])


def test_cache_features_takes_the_thermal_input_arm() -> None:
    base = ["cache-features", "siglip2-base-naflex", "--camera", "thermal"]

    assert cli.build_parser().parse_args(base).thermal_input == "replicate"
    args = cli.build_parser().parse_args([*base, "--thermal-input", "equalise"])
    assert args.thermal_input == "equalise"


def test_train_head_stem_dispatches_to_the_stem_runner(monkeypatch) -> None:
    from aerial_search.experiments import stem_experiment

    seen = {}
    monkeypatch.setattr(
        stem_experiment, "run_stem_head", lambda **kw: seen.update(kw) or {"test": {}}
    )
    command = ["train-head", "thermal", "--fold", "F", "--percent", "10"]
    command += ["--input-handling", "stem", "--steps", "9", "--micro-batch", "4"]

    cli.main([*command, "--scratch", "--device", "cpu"])

    assert seen["recipe"].steps == 9 and seen["recipe"].micro_batch == 4
    assert seen["scratch"] is True and seen["camera"] == "thermal"
