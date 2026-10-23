"""Commands that live in a project package, run through `aerial-search` (#160)."""

import argparse
import importlib
import sys
import types

import pytest

from aerial_search import cli


def test_every_registered_project_module_imports_and_has_a_main() -> None:
    for name, command in cli.PROJECT_COMMANDS.items():
        module = importlib.import_module(command.module)
        assert callable(module.main), name


def test_no_project_command_shadows_a_shared_one(monkeypatch) -> None:
    cli.build_parser()  # the real registry
    monkeypatch.setitem(cli.PROJECT_COMMANDS, "prepare", cli.ProjectCommand("x", ""))
    with pytest.raises(argparse.ArgumentError, match="prepare"):
        cli.build_parser()


@pytest.fixture
def fake(monkeypatch) -> list[list[str]]:
    seen: list[list[str]] = []
    module = types.ModuleType("fake_project_cli")
    module.__dict__["main"] = seen.append
    monkeypatch.setitem(sys.modules, "fake_project_cli", module)
    monkeypatch.setitem(
        cli.PROJECT_COMMANDS,
        "train-fake",
        cli.ProjectCommand("fake_project_cli", "train the fake approach"),
    )
    return seen


def test_a_project_command_gets_its_whole_command_line(fake) -> None:
    # one project module can serve several commands, as a parser of its own
    cli.main(["train-fake", "--fold", "220109_Baker", "--help"])
    assert fake == [["train-fake", "--fold", "220109_Baker", "--help"]]


def test_project_commands_are_listed_in_the_help(fake, capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "train-fake" in capsys.readouterr().out


def test_a_project_module_is_imported_only_when_its_command_runs(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "lazy_probe.py").write_text("def main(argv):\n    return argv\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "lazy_probe", raising=False)
    monkeypatch.setitem(
        cli.PROJECT_COMMANDS, "train-probe", cli.ProjectCommand("lazy_probe", "")
    )
    cli.build_parser()
    assert "lazy_probe" not in sys.modules
    cli.main(["train-probe", "--x"])
    assert "lazy_probe" in sys.modules
    monkeypatch.delitem(sys.modules, "lazy_probe")


def test_a_missing_project_module_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.setitem(
        cli.PROJECT_COMMANDS, "train-gone", cli.ProjectCommand("no_such_project", "")
    )
    with pytest.raises(SystemExit, match="no_such_project"):
        cli.main(["train-gone"])
