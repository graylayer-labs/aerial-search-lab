# aerial-search-lab

Process (roles, board, approvals, updates, limits, review, commits): see
~/.claude/rules/kitchen.md and the playbooks in ~/.claude/skills/.

A lab of approaches that detect people in drone RGB and thermal footage with
few labels, all scored by one shared evaluation. Read the intent first; it
overrides anything below that conflicts with it.

@INTENT.md

## Where to look

Open the row you need; do not list folders and guess.

| You need | Open |
|---|---|
| Which folder does what, where to start | `docs/knowledge/map.md` |
| How to run or change things: setup, data, experiments, cloud runs, diagrams, signing | `docs/knowledge/recipes.md` |
| Why it is built this way | `docs/knowledge/decisions.md` (points to `docs/decisions.md`) |
| Traps (zsh, gh, worktrees, CI, AWS names) | `docs/knowledge/gotchas.md` |
| What changed lately | `docs/progress.md`, `CHANGELOG.md` |
| Evidence behind a decision | `docs/<topic>-review.md` (pairing, site folds, metric) |
| What a milestone found | `docs/milestones/<name>.md` |
| How it was built, step by step | `docs/guide/` (raw notes in `docs/guide/notes/<issue>.md`) |

## Commands

```bash
uv sync --dev          # install the workspace; a fresh worktree has no environment
uv run pytest          # tests of shared/ and projects/
uv run ruff check --fix . && uv run ruff format . && uv run ty check
```

CI runs these four checks. More in `recipes.md`.

## Board

- Board: https://github.com/orgs/graylayer-labs/projects/1 (project number 1,
  owner `graylayer-labs`, public). Issues:
  https://github.com/graylayer-labs/aerial-search-lab/issues
- Project id `PVT_kwDOEDMalc4BlE5m`; Status field id
  `PVTSSF_lADOEDMalc4BlE5mzhjzG-Q`; options Todo `f75ad846`, In Progress
  `47fc9ee4`, Done `98236657`. Status lives on the board only.
- Epics are parent issues labelled `epic`, one per milestone; their sub-issues
  are the tasks. The epic is the milestone (GitHub's milestone field is
  unused). Only the current epic has sub-issues; follow its "Order of work".
  An epic row stays In Progress while any task is open. Work that crosses
  milestones is a standalone issue with no parent.
- Labels, standing issues and board commands: `recipes.md`, Work the board.

## Repo rules

- `main` is protected: a PR merges only when the `checks` and `gate` CI jobs
  pass (`gate` is the org check: title, issue link, size under 400 lines,
  protected paths), and merging goes through GitHub, never locally
  (`gh pr merge <n> --auto --squash`). Commits are signed locally (setup in
  `recipes.md`), but main no longer requires verified signatures: agent
  commits cannot verify yet (owner, 2026-11-14, #215).
- Every PR closes an issue, except Dependabot PRs: the lead merges those when
  `checks` and `gate` pass.
- Logic under `shared/` or `projects/*/src/`, experiments and metrics are
  checked by the `reviewer` agent before auto-merge; docs and config are read
  by the lead. The PR says which, and its Evidence section holds the pasted
  output of the lead's own re-run of the checks.
- Project agents and skills live in `.claude/agents/` and `.claude/skills/`
  (`/tidy`, `/audit`, `/coldstart`, `/guide`, `/blog`). A rule change here is
  carried into the file that implements it, in the same PR.
- Opus only where a wrong answer corrupts results: pairing, splits, metrics
  and their reviews.
- Walls in `.claude/settings.json`: no force-push or push to `main`; no
  deletion under `data/raw/` or in S3 (`aws s3 rm|rb|sync --delete` denied);
  no `npm install`; Archify `brands` and `preview` denied (no brand URLs,
  never `--open`); `aws *`, `gh repo edit|delete`, `gh release` ask the owner.
  Publishing a blog post, release or repo setting is the owner's.

## Data

`data/` is git-ignored, so a fresh clone or worktree has none.

| What | Where |
|---|---|
| Raw WiSARD images and labels, 41 GB | `data/raw/wisard-full/` in the main checkout |
| Manifests, regenerable | `data/manifests/wisard-full/` in the main checkout |
| Durable copy | S3 bucket `ssl-aerial-person-detection-data-eu-west1`, `eu-west-1`, under `wisard/raw/wisard-full/`; versioning on; checked equal to local by path and size on 2026-09-29 |

- The main checkout is the first path of `git worktree list`. Agent worktrees
  get `data/` as a symlink to it; if missing, read it by absolute path.
- Data you cannot reach: stop and say so. Never invent figures or regenerate
  from nothing. A missing local directory comes from S3 (`fetch-directory`),
  not another copy.
- Pinned by `shared/aerial_search/src/aerial_search/data/checksums/wisard-full.tsv.gz`
  (100,794 files, path, size, SHA-256); root hash
  `6e5e5d554c4be119f15bb636049800788904f3a2e2759cabce402c676d1575ed`.
- A normal `prepare` or run (`train-ssl`, `train-detector`, `train-head`)
  verifies its directories against the list and refuses missing, scratch or
  unverified manifests; `run.json` records the data it was verified against.
  `--scratch` skips the check and records `verified: false`. Commands and
  details: `recipes.md`. Pairing evidence: `docs/wisard-pairing-review.md`.

## AWS and compute

- Profile `ssl-aerial` (`AWS_PROFILE` in `.claude/settings.json`) assumes role
  `ssl-aerial-person-detection-agent`: list, read and write objects in the
  project bucket, nothing else; no deletes, no bucket changes. Never name
  another profile (the owner's sign-in is administrator). Fresh machine:
  `recipes.md`, "Set up the AWS profile". Expired token: ask the owner to
  sign in again. The AWS names keep the repo's old name on purpose (#160).
- Cloud GPU: `aerial-search sagemaker-train`, `ml.g4dn.xlarge`, managed spot
  with checkpoints, EUR 15 per run and EUR 60 per calendar month (owner,
  2026-10-14 #109, confirmed #125). Spot only: AWS declined on-demand
  (#115). Ask `aws-manager` for month-to-date spend first.
- Cloud CPU: `sagemaker-train-head`, `ml.m5.xlarge` spot, 2 h per job, EUR 5
  per month (owner, 2026-10-23, #184). Details and guard: `recipes.md`.
- Laptop (M4, 24 GB): size runs under 30 minutes; a run over one hour goes to
  the cloud within the limits above.

## Experiments

- A quoted result comes from a commit on `main` with a clean tree. A run from
  a branch or dirty tree is scratch (`--scratch`): debug only, never quote
  its numbers. `start_run` (`shared/aerial_search/src/aerial_search/run.py`)
  enforces this, fails closed, never fetches (run `git fetch` first), and
  never overwrites a run directory.
- Every run has a committed config and a seed and writes `outputs/<run-name>/`
  with `run.json` (commit, config, seed, machine, input SHA-256s). It is
  quotable only if `"status": "completed"`; `started` means crashed or still
  running, `failed` records the error. Exceptions to "the run directory is the
  only place a run writes": the feature cache and `--checkpoint-dir`.
- Any number in a doc or README must be reproducible from a committed config;
  if it cannot be traced, remove it. Report negative results: what was
  expected, what happened, the likely reason.
- Compare against a baseline on a held-out split. Splits are by site-day
  (every clip from one site on one date), never by clip or frame: neighbouring
  frames are near-duplicates. Each labelled site-day is the test set of one
  fold. Report per fold and as mean and spread, never one pooled number
  (`docs/site-folds-review.md`).
- Every table or figure showing `ap_iou25` shows `ap_iou50` beside it
  (`docs/evaluation-metric-review.md`).
- Test what would silently corrupt results: pairing, splits, label handling,
  metrics. Skip tests that restate the implementation.

## Documentation

One home per fact: why `INTENT.md`; plans, the board; decisions
`docs/decisions.md`; evidence `docs/<topic>-review.md`; guide notes
`docs/guide/notes/<issue>.md`, deleted once in a chapter. Changelog rules:
`recipes.md`, "Write a changelog entry".
