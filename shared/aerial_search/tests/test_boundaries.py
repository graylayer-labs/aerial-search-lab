"""The shared package never imports a project (#160).

Every approach depends on `aerial_search`; if it imported an approach, the
evaluation would no longer be shared. Projects are named only as strings in
`cli.PROJECT_COMMANDS`, which this scan allows.
"""

import ast
from pathlib import Path

from aerial_search import REPO_ROOT
from aerial_search.cloud import entry

SHARED = REPO_ROOT / "shared" / "aerial_search" / "src" / "aerial_search"


def project_packages() -> set[str]:
    """The import names of every workspace member under projects/."""
    return {
        package.name
        for src in entry.source_dirs(REPO_ROOT)
        if src.parent.parent.name == "projects"
        for package in src.iterdir()
        if (package / "__init__.py").is_file()
    }


def imported(path: Path) -> set[str]:
    """The top-level packages a module imports, wherever the import sits."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_the_scan_knows_the_projects() -> None:
    assert {"finetuned_detector", "frozen_head"} <= project_packages()


def test_no_shared_module_imports_a_project() -> None:
    projects = project_packages()
    found = {
        str(path.relative_to(SHARED)): sorted(imported(path) & projects)
        for path in SHARED.rglob("*.py")
        if imported(path) & projects
    }
    assert found == {}
