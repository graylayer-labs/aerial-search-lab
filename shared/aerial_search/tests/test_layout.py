"""Every lookup of the repository root lands on the workspace root (#160).

The package sits four levels below the root, in shared/aerial_search/src/. A
lookup that counts parents wrongly does not fail loudly: the AWS profile falls
back to the default credentials, or a cloud job hashes the wrong directory.
"""

import importlib.util
from pathlib import Path

from aerial_search import run
from aerial_search.cloud import entry, source
from aerial_search.data import checksums


def workspace_root() -> Path:
    """Found independently of the code under test: the directory with uv.lock."""
    here = Path(__file__).resolve()
    return next(p for p in here.parents if (p / "uv.lock").is_file())


def test_run_finds_the_root_from_the_running_code() -> None:
    assert run.archive_root(run._code_location()) == workspace_root()


def test_the_job_entry_code_root_is_the_workspace_root() -> None:
    assert workspace_root() == entry.CODE_ROOT


def test_the_settings_file_is_the_repositorys_own() -> None:
    assert workspace_root() / ".claude" / "settings.json" == checksums.SETTINGS
    assert checksums.SETTINGS.is_file()


def test_the_sagemaker_entry_script_puts_the_shared_package_on_the_path() -> None:
    script = workspace_root() / "tools" / "sagemaker_entry.py"
    spec = importlib.util.spec_from_file_location("sagemaker_entry", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert workspace_root() / "shared" / "aerial_search" / "src" == module.SOURCE
    assert (module.SOURCE / "aerial_search" / "__init__.py").is_file()


def test_the_job_puts_every_member_source_on_the_path(tmp_path: Path) -> None:
    for member in ("shared/aerial_search", "projects/b", "projects/_template"):
        (tmp_path / member / "src").mkdir(parents=True)
        (tmp_path / member / "pyproject.toml").write_text("[project]\n")
    assert entry.source_dirs(tmp_path) == [
        tmp_path / "shared" / "aerial_search" / "src",
        tmp_path / "projects" / "b" / "src",
    ]


def test_the_constraints_name_no_workspace_member() -> None:
    # pip refuses an editable requirement as a constraint
    lines = source.export_constraints(workspace_root()).decode().splitlines()
    assert not [line for line in lines if line.startswith("-e")]
    assert any(line.startswith("pycocotools==") for line in lines)
