# Map

Where things are in this repo, so a session opens the right file first.

## Layout

```
pyproject.toml       uv workspace root; not a package itself
shared/aerial_search/  the shared evaluation, package aerial_search: data/
                     (pairing, folds, records), evaluation/ (metrics, tables),
                     run.py, runner.py, cloud/, the aerial-search CLI;
                     and its tests/
projects/<name>/     one package per approach, with its README and tests/
projects/_template/  the starting point for a new approach
docs/                milestone write-ups and decision notes
tools/               one-off scripts that are worth keeping
data/                datasets and manifests (git-ignored)
outputs/             run artefacts (git-ignored)
```

## Entries

| Path | What lives there | Open first |
|---|---|---|
| `shared/aerial_search/src/aerial_search/data/wisard.py` | Pairing of RGB and thermal frames; `WISARD_COLLECTIONS` | `docs/wisard-pairing-review.md` |
| `shared/aerial_search/src/aerial_search/data/checksums/` | The pinned dataset list, `wisard-full.tsv.gz` | `CLAUDE.md` (Data) |
| `shared/aerial_search/src/aerial_search/run.py` | `start_run`: clean-main check, `run.json` | `CLAUDE.md` (Experiments) |
| `shared/aerial_search/src/aerial_search/runner.py` | What every experiment repeats: `experiment_run` (start and finish a run), `run_name`, `write_training`, `score`, `batch_order`, `train_best_step` | the module docstring |
| `projects/*/tests/fixtures/golden/`, `projects/conftest.py` | Pinned outputs of whole runs; `UPDATE_GOLDEN=1` rewrites them, only for an intended change | `projects/conftest.py` |
| `shared/aerial_search/src/aerial_search/cloud/` | SageMaker launcher, cost guard, job entry | `cloud/entry.py` |
| `shared/aerial_search/tests/fixtures/gpu_request.json` | Pins the GPU job request | `recipes.md` (cloud run) |
| `projects/finetuned-detector/` | The bar: a fine-tuned detector | its `README.md` |
| `projects/frozen-head/` | A detection head on frozen features, feature cache | its `README.md`, `docs/detection-head-design.md` |
| `projects/_template/` | Starting point for an approach | `projects/README.md` |
| `tools/` | Kept one-off scripts (`install_archify.sh`, `pairing_contact_sheets.py`, `sagemaker_entry.py`, ...) | the script |
| `docs/milestones/` | What each milestone found | the newest file |
| `docs/guide/` | Step-by-step build guide; `notes/<issue>.md` is raw material until in a chapter | `docs/guide/README.md` |
| `docs/blog/`, `docs/figures/` | Blog drafts; Archify figures and source JSON | |
| `docs/progress.md`, `CHANGELOG.md` | What changed lately | |
| `data/`, `outputs/` | Git-ignored; written by code, never edited by hand. `data/` is a symlink to the main checkout's copy in agent worktrees | |

## Projects

- Each approach to the problem is one project under `projects/`.
- It shares the evaluation: data, folds, frame selection
  (`aerial_search.data.records`), metrics and `start_run` come from
  `aerial_search`, and the project changes none of them.
- Its commands are registered in `PROJECT_COMMANDS` in the shared CLI, so
  they run as `aerial-search <command>`, in a cloud job too.
- `shared/` never imports a project; a test enforces it.
