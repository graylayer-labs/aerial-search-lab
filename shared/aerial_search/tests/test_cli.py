import json
from pathlib import Path

import pytest

from aerial_search.cli import build_parser


def test_fetch_command_parses() -> None:
    args = build_parser().parse_args(["fetch", "wisard-sample", "--no-extract"])

    assert args.command == "fetch"
    assert args.dataset == "wisard-sample"
    assert args.no_extract is True


def test_prepare_command_parses() -> None:
    args = build_parser().parse_args(["prepare", "data/raw/wisard-sample"])

    assert args.command == "prepare"
    assert args.collection is None


def test_prepare_takes_an_explicit_subset_of_collections() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "data/raw/wisard-full",
            "--collection",
            "220109_Baker_Enterprise_1",
            "--collection",
            "210417_MtErie_Enterprise_0005",
        ]
    )

    assert args.collection == [
        "220109_Baker_Enterprise_1",
        "210417_MtErie_Enterprise_0005",
    ]


def test_folds_commands_take_the_raw_images_and_the_manifests() -> None:
    parser = build_parser()
    for command in ("folds", "check-folds"):
        args = parser.parse_args(
            [command, "data/raw/wisard-full", "data/manifests/wisard-full"]
        )
        assert args.command == command
        assert str(args.source) == "data/raw/wisard-full"
        assert str(args.manifests) == "data/manifests/wisard-full"


def test_cache_features_takes_a_model_a_camera_and_optional_collections() -> None:
    args = build_parser().parse_args(
        [
            "cache-features",
            "siglip2-base-naflex",
            "--camera",
            "thermal",
            "--collection",
            "220109_Baker_Enterprise_1",
        ]
    )

    assert args.command == "cache-features"
    assert args.camera == "thermal"
    assert args.collection == ["220109_Baker_Enterprise_1"]
    assert (args.token_budget, args.pooling) == (1024, 2)
    assert str(args.manifests) == "data/manifests/wisard-full"
    again = build_parser().parse_args(
        ["cache-features", "siglip2-base-naflex", "--camera", "rgb", "--part", "2/3"]
    )
    assert again.part == (2, 3)
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "cache-features",
                "siglip2-base-naflex",
                "--camera",
                "rgb",
                "--part",
                "4/3",
            ]
        )
    with pytest.raises(SystemExit):  # one camera per session
        build_parser().parse_args(["cache-features", "siglip2-base-naflex"])


def test_experiments_need_a_fold() -> None:
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-ssl"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-detector", "rgb", "scratch"])


def test_detection_command_parses() -> None:
    args = build_parser().parse_args(
        ["train-detector", "thermal", "ssl", "--fold", "220109_Baker"]
    )

    assert args.command == "train-detector"
    assert args.modality == "thermal"
    assert args.initialization == "ssl"


def test_experiment_commands_accept_scratch_flag() -> None:
    parser = build_parser()
    assert (
        parser.parse_args(["train-ssl", "--fold", "220109_Baker", "--scratch"]).scratch
        is True
    )
    assert parser.parse_args(["train-ssl", "--fold", "220109_Baker"]).scratch is False
    assert parser.parse_args(
        ["train-detector", "rgb", "scratch", "--fold", "220109_Baker", "--scratch"]
    ).scratch


def test_experiment_refuses_to_start_outside_a_git_repo(tmp_path, monkeypatch) -> None:
    import pytest

    from aerial_search.cli import main

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="refusing to start"):
        main(["train-ssl", "--fold", "220109_Baker"])
    assert not (tmp_path / "outputs").exists()


def _fake_repo(tmp_path, monkeypatch):
    import subprocess

    from aerial_search import run as run_module

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("outputs/\n")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*g, "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*g, "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run([*g, "commit", "-q", "-m", "i"], cwd=repo, check=True)
    subprocess.run(
        [*g, "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=repo, check=True
    )
    monkeypatch.setattr(run_module, "_code_location", lambda: repo)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "data_quality.json").write_text(
        json.dumps(
            {
                "provenance": {"commit": "abc", "scratch": False},
                "data": {"verified": True, "directories": [], "root_hash": "r"},
            }
        )
    )
    view = manifests / "folds" / "220109_Baker" / "paired"
    view.mkdir(parents=True)
    (view / "train_100pct.jsonl").write_text("{}\n")
    (view / "validation.jsonl").write_text("{}\n")
    return repo, manifests


def test_output_option_is_gone() -> None:
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-ssl", "--output", "x"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-detector", "rgb", "scratch", "--output", "x"])


def test_detector_calls_start_run_before_training(monkeypatch) -> None:
    import pytest

    from aerial_search import cli
    from aerial_search import run as run_module
    from aerial_search.experiments import detection_experiment

    calls: list[str] = []

    def refuse(*args, **kwargs):
        calls.append("start_run")
        raise run_module.ProvenanceError("no")

    def train(*args, **kwargs):
        calls.append("train")

    monkeypatch.setattr(run_module, "start_run", refuse)
    monkeypatch.setattr(detection_experiment, "run_detection_experiment", train)
    with pytest.raises(SystemExit, match="refusing to start"):
        cli.main(["train-detector", "rgb", "scratch", "--fold", "220109_Baker"])
    assert calls == ["start_run"]


def test_ssl_run_writes_into_run_dir_and_completes(tmp_path, monkeypatch) -> None:
    import json
    from dataclasses import dataclass

    from aerial_search import cli
    from aerial_search.experiments import ssl_experiment

    repo, manifests = _fake_repo(tmp_path, monkeypatch)
    seen = {}

    @dataclass
    class Result:
        top1: float = 0.5

    def fake(data_root, manifests, output, **kwargs):
        seen["output"] = output
        seen["manifests"] = manifests
        seen["seed"] = kwargs["seed"]
        return Result()

    monkeypatch.setattr(ssl_experiment, "run_experiment", fake)
    cli.main(
        [
            "train-ssl",
            "--manifests",
            str(manifests),
            "--fold",
            "220109_Baker",
            "--run-name",
            "a",
        ]
    )
    assert seen["output"] == repo / "outputs" / "a"
    data = json.loads((seen["output"] / "run.json").read_text())
    assert data["status"] == "completed"
    assert data["seed"] == seen["seed"]
    assert (data["fold"], data["view"]) == ("220109_Baker", "paired")
    assert {i["path"] for i in data["inputs"]} == {
        str(manifests / "folds" / "220109_Baker" / "paired" / "train_100pct.jsonl"),
        str(manifests / "folds" / "220109_Baker" / "paired" / "validation.jsonl"),
    }
    assert seen["manifests"] == manifests / "folds" / "220109_Baker" / "paired"


def test_failed_experiment_is_marked_failed(tmp_path, monkeypatch) -> None:
    import json

    import pytest

    from aerial_search import cli
    from aerial_search.experiments import ssl_experiment

    repo, manifests = _fake_repo(tmp_path, monkeypatch)

    def crash(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(ssl_experiment, "run_experiment", crash)
    with pytest.raises(RuntimeError):
        cli.main(
            [
                "train-ssl",
                "--manifests",
                str(manifests),
                "--fold",
                "220109_Baker",
                "--run-name",
                "a",
            ]
        )
    data = json.loads((repo / "outputs" / "a" / "run.json").read_text())
    assert data["status"] == "failed"
    assert data["error"]["type"] == "RuntimeError"


def test_detector_gets_seed_and_run_dir(tmp_path, monkeypatch) -> None:
    from dataclasses import dataclass

    from aerial_search import cli
    from aerial_search.experiments import detection_experiment

    repo, manifests = _fake_repo(tmp_path, monkeypatch)
    seen = {}

    @dataclass
    class Result:
        recall: float = 0.1

    def fake(data_root, manifests, output, **kwargs):
        seen.update(output=output, **kwargs)
        return Result()

    monkeypatch.setattr(detection_experiment, "run_detection_experiment", fake)
    cli.main(
        [
            "train-detector",
            "rgb",
            "scratch",
            "--manifests",
            str(manifests),
            "--fold",
            "220109_Baker",
        ]
    )
    assert seen["output"] == repo / "outputs" / "detector-rgb-scratch"
    assert seen["seed"] == cli.SEED


def test_detector_refuses_a_checkpoint_pretrained_in_another_fold(
    tmp_path, monkeypatch
) -> None:
    import json

    import pytest

    from aerial_search import cli
    from aerial_search.experiments import detection_experiment

    repo, manifests = _fake_repo(tmp_path, monkeypatch)
    parent = tmp_path / "ssl-mterie"
    parent.mkdir()
    (parent / "model.pt").write_text("weights")
    (parent / "run.json").write_text(
        json.dumps(
            {
                "run_name": "ssl-mterie",
                "scratch": False,
                "commit": "abc",
                "status": "completed",
                "fold": "210417_MtErie",
                "view": "paired",
            }
        )
    )
    monkeypatch.setattr(
        detection_experiment, "run_detection_experiment", lambda *a, **k: None
    )
    with pytest.raises(SystemExit, match="210417_MtErie.*220109_Baker"):
        cli.main(
            [
                "train-detector",
                "rgb",
                "ssl",
                "--manifests",
                str(manifests),
                "--fold",
                "220109_Baker",
                "--ssl-checkpoint",
                str(parent / "model.pt"),
            ]
        )
    assert not (repo / "outputs").exists()


def _check_folds_output(monkeypatch, capsys, report) -> tuple[str, int | str | None]:
    from aerial_search import cli

    monkeypatch.setattr(cli, "check_folds", lambda *a, **k: report)
    monkeypatch.setattr(cli, "_image_size", lambda source: None)
    code = 0
    try:
        cli.main(["check-folds", "raw", "manifests"])
    except SystemExit as exit_:
        code = exit_.code
    return capsys.readouterr().out, code


def test_check_folds_marks_every_line_of_a_passing_check(monkeypatch, capsys) -> None:
    from aerial_search.data.folds import CheckReport

    out, code = _check_folds_output(
        monkeypatch, capsys, CheckReport(lines=["no leaks", "gap 250"])
    )

    assert code == 0
    assert out.splitlines() == ["OK   no leaks", "OK   gap 250"]


def test_check_folds_prints_only_the_failures_of_a_failing_check(
    monkeypatch, capsys
) -> None:
    from aerial_search.data.folds import CheckReport

    report = CheckReport(lines=["no leaks", "gap 250"], problems=["train holds test"])

    out, code = _check_folds_output(monkeypatch, capsys, report)

    assert code == 1
    assert out.splitlines() == ["FAIL train holds test"]


def test_checksum_then_check_data_reports_every_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aerial_search.cli import main

    source = tmp_path / "raw"
    (source / "a_0001").mkdir(parents=True)
    (source / "a_0001" / "x.jpg").write_bytes(b"xx")
    (source / "a_0001" / "y.jpg").write_bytes(b"yy")
    pinned = tmp_path / "pin.tsv.gz"

    main(["checksum", str(source), "--output", str(pinned)])
    main(["check-data", str(source), "--list", str(pinned)])
    assert "OK" in capsys.readouterr().out

    (source / "a_0001" / "x.jpg").write_bytes(b"XX")
    (source / "a_0001" / "y.jpg").unlink()
    (source / "a_0001" / "z.jpg").write_bytes(b"z")
    with pytest.raises(SystemExit) as raised:
        main(["check-data", str(source), "--list", str(pinned)])
    assert raised.value.code == 1
    out = capsys.readouterr().out
    assert "changed a_0001/x.jpg" in out
    assert "missing a_0001/y.jpg" in out
    assert "added   a_0001/z.jpg" in out


def _strict_code_state(*, scratch: bool, repo: Path | None = None) -> dict:
    """Refuses a normal run, as a dirty tree would; allows scratch."""
    from aerial_search import run

    if not scratch:
        raise run.ProvenanceError("the working tree has uncommitted changes")
    return {"commit": "abc", "dirty": " M x"}


def test_prepare_refuses_a_dirty_tree_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aerial_search.cli as cli
    from aerial_search import run

    monkeypatch.setattr(run, "code_state", _strict_code_state)
    monkeypatch.setattr(
        cli, "prepare_manifests", lambda *a, **k: pytest.fail("must not prepare")
    )
    with pytest.raises(SystemExit, match="uncommitted"):
        cli.main(["prepare", str(tmp_path), "--output", str(tmp_path / "out")])


def test_prepare_scratch_records_scratch_and_skips_the_data_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aerial_search.cli as cli
    from aerial_search import run

    monkeypatch.setattr(run, "code_state", _strict_code_state)
    seen: dict = {}

    def fake_prepare(*args: object, **kwargs: object) -> dict[str, int]:
        seen.update(kwargs)
        return {"all_pairs": 0}

    monkeypatch.setattr(cli, "prepare_manifests", fake_prepare)
    cli.main(["prepare", str(tmp_path), "--scratch"])
    assert seen["provenance"] == {"commit": "abc", "scratch": True, "dirty": " M x"}
    assert seen["verify"] is False and seen["force"] is False


def test_normal_prepare_verifies_and_passes_scratch_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aerial_search.cli as cli
    from aerial_search import run

    monkeypatch.setattr(
        run, "code_state", lambda *, scratch, repo=None: {"commit": "abc"}
    )
    seen: dict = {}
    monkeypatch.setattr(
        cli, "prepare_manifests", lambda *a, **k: seen.update(k) or {"all_pairs": 0}
    )
    cli.main(["prepare", str(tmp_path)])
    assert seen["provenance"] == {"commit": "abc", "scratch": False}
    assert seen["verify"] is True


def test_checksum_refuses_to_overwrite_a_list_without_force(tmp_path: Path) -> None:
    from aerial_search.cli import main

    source = tmp_path / "raw"
    (source / "a").mkdir(parents=True)
    (source / "a" / "x").write_bytes(b"x")
    pinned = tmp_path / "pin.tsv.gz"
    main(["checksum", str(source), "--output", str(pinned)])
    with pytest.raises(SystemExit, match="--force"):
        main(["checksum", str(source), "--output", str(pinned)])
    main(["checksum", str(source), "--output", str(pinned), "--force"])


def test_fetch_directory_prints_the_aws_error_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aerial_search.cli as cli
    from aerial_search.data import checksums

    def fail(*args: object, **kwargs: object) -> Path:
        raise checksums.FetchError("aws failed (exit 255): token expired")

    monkeypatch.setattr(checksums, "ensure_directory", fail)
    with pytest.raises(SystemExit, match="token expired"):
        cli.main(["fetch-directory", "x", "--data-root", str(tmp_path)])


def test_train_head_takes_a_camera_fold_fraction_and_recipe(monkeypatch) -> None:
    from aerial_search import cli
    from aerial_search.experiments import head_experiment

    args = build_parser().parse_args(
        ["train-head", "thermal", "--fold", "220109_Baker", "--percent", "5"]
    )
    assert (args.camera, args.fold, args.percent, args.seed) == (
        "thermal",
        "220109_Baker",
        5,
        7,
    )
    assert args.cache == Path("outputs/features/siglip2-base-naflex-1024tok")
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["train-head", "rgb", "--fold", "x", "--percent", "3"]
        )

    seen = {}
    monkeypatch.setattr(
        head_experiment, "run_head", lambda **kw: seen.update(kw) or {"test": {}}
    )
    command = ["train-head", "rgb", "--fold", "F", "--percent", "1", "--steps", "9"]
    cli.main([*command, "--scratch", "--device", "cpu"])
    assert seen["recipe"].steps == 9 and seen["scratch"] is True
    assert seen["percent"] == 1 and seen["camera"] == "rgb"


def test_head_table_prints_the_table_of_a_runs_directory(tmp_path, capsys) -> None:
    from aerial_search import cli

    run = tmp_path / "rgb-A-100"
    run.mkdir()
    config = {"camera": "rgb", "fold": "A", "percent": 100}
    record = {"status": "completed", "scratch": False, "config": config}
    (run / "run.json").write_text(json.dumps(record))
    overall = {"ap_iou25": 0.2, "ap_iou50": 0.1, "recall_at_fppi": {}}
    metrics = {"overall": overall, "by_size": {}}
    (run / "detection_metrics.json").write_text(json.dumps(metrics))

    cli.main(["head-table", str(tmp_path), "--json", str(tmp_path / "t.json")])
    assert "0.200 / 0.100" in capsys.readouterr().out
    assert json.loads((tmp_path / "t.json").read_text())["rgb"]["100"]["n_folds"] == 1
