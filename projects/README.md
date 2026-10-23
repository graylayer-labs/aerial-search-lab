# Projects

One folder per approach to finding people in aerial RGB and thermal frames.
Each is its own uv workspace package. All of them depend on the shared
`aerial-search` package in `shared/`, which holds the data, the site-day
folds, the metrics and run provenance, so every approach is scored the same
way.

```
projects/<name>/
├── pyproject.toml   # depends on aerial-search through the workspace
├── README.md        # Goal, Approach, Expected results, Run, Results, Status
├── src/<name>/      # code; cli.py holds the approach's commands
└── tests/           # tests
```

## Starting a new approach

1. Copy the template: `cp -R projects/_template projects/<name>`.
2. Rename the package: in its `pyproject.toml` set `name = "<name>"` and
   `packages = ["src/<name_underscored>"]`, then
   `mv src/my_approach src/<name_underscored>` and fix the imports in
   `src/<name_underscored>/cli.py` and `tests/test_smoke.py`.
3. Add it to the root `pyproject.toml`: the `dev` dependency group,
   `[tool.uv.sources]` (`<name> = { workspace = true }`) and
   `[tool.coverage.run] source_pkgs`.
4. Register each command in `PROJECT_COMMANDS` in
   `shared/aerial_search/src/aerial_search/cli.py`, naming `<name_underscored>.cli`.
5. Fill in the README's Goal, Approach and Expected results before the first
   run.
6. `uv lock`, `uv sync --dev`, then the four checks, and commit with the
   updated `uv.lock`.
7. Add the approach to the table below.

Third-party dependencies go in the project's own `pyproject.toml`, added
with `uv add --package <name> <dependency>`.

## Approaches

| Project | Approach | Status |
|---|---|---|
| [finetuned-detector](finetuned-detector/) | COCO-pretrained detector fine-tuned on few labels: the bar | Results at the cache's input size; tiled runs next |
| [frozen-head](frozen-head/) | Person-centre head on frozen SigLIP 2 features | Results in; below the bar where labels are scarce |
