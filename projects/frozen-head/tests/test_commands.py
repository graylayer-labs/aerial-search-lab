"""The frozen head's commands, run through `aerial-search` (#160)."""

import importlib.machinery

import pytest

from aerial_search import REPO_ROOT, cli
from aerial_search.cloud import entry
from frozen_head import cli as project
from frozen_head.feature_cache import Summary

COMMANDS = ("cache-features", "check-cache")


def test_the_commands_reach_this_project_in_the_job() -> None:
    # In the job nothing is installed: only the members' src/ are on sys.path.
    paths = [str(d) for d in entry.source_dirs(REPO_ROOT)]
    for name in COMMANDS:
        assert cli.PROJECT_COMMANDS[name].module == "frozen_head.cli"
    assert importlib.machinery.PathFinder.find_spec("frozen_head", paths)


def test_cache_features_takes_a_model_a_camera_and_optional_collections() -> None:
    args = project.build_parser().parse_args(
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
    again = project.build_parser().parse_args(
        ["cache-features", "siglip2-base-naflex", "--camera", "rgb", "--part", "2/3"]
    )
    assert again.part == (2, 3)
    with pytest.raises(SystemExit):
        project.build_parser().parse_args(
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
        project.build_parser().parse_args(["cache-features", "siglip2-base-naflex"])


def test_check_cache_lists_missing_frames_and_exits_1(monkeypatch, capsys) -> None:
    from frozen_head import feature_cache

    seen = {}

    def missing_frames(cache, manifests, cameras):
        seen["cameras"] = cameras
        return {"thermal": ["a.jpg"]}

    monkeypatch.setattr(feature_cache, "missing_frames", missing_frames)
    with pytest.raises(SystemExit) as failed:
        cli.main(["check-cache", "c", "m", "--camera", "thermal"])
    assert failed.value.code == 1 and seen["cameras"] == ("thermal",)
    assert capsys.readouterr().out == "thermal: 1 missing\n  a.jpg\n"


def test_cache_features_passes_its_flags_and_refuses_clearly(monkeypatch) -> None:
    from frozen_head import feature_cache

    seen = {}

    def cache_features(**kw):
        seen.update(kw)
        raise feature_cache.CacheError("settings differ")

    monkeypatch.setattr(feature_cache, "cache_features", cache_features)
    command = ["cache-features", "siglip2-base-naflex", "--camera", "rgb"]
    with pytest.raises(SystemExit, match="refusing to cache: settings differ"):
        cli.main([*command, "--part", "1/2", "--scratch"])
    assert seen["part"] == (1, 2) and seen["scratch"] is True
    assert seen["camera"] == "rgb" and seen["input_handling"] == "replicate"
    monkeypatch.setattr(feature_cache, "cache_features", lambda **kw: Summary())
    cli.main(command)
