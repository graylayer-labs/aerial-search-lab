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
- Labels: `agent:fable|opus|sonnet|haiku`, `type:bug|enhancement|documentation|
  experiment`, `epic`, `needs-owner`, `tooling-candidate`. The org template's
  `data:`, `model:`, `source:`, `status:` labels are unused.
- Two standing issues never close: "Reusable tooling candidates" (#9) and
  "Audit log" (#24). A step done by hand that others would repeat: comment on #9.
- Commands to attach sub-issues, read a task and move an item:
  `recipes.md`.

## Repo rules

- `main` is protected: a PR merges only when the `checks` and `gate` CI jobs
  pass (`gate` is the org check: title, issue link, size under 400 lines,
  protected paths), every commit is signed (setup in `recipes.md`), and merging goes through GitHub,
  never locally (`gh pr merge <n> --auto --squash`).
- Every PR closes an issue, except Dependabot PRs: the lead merges those when
  `checks` and `gate` pass.
- Logic under any `src/`, experiments and metrics are checked by the
  `reviewer` agent before auto-merge; docs and config are read by the lead.
  The PR says which, and its Evidence section holds the pasted output of the
  lead's own re-run of the checks.
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
  another profile (the owner's sign-in is administrator). A fresh machine
  needs the profile in `~/.aws/config`: `role_arn` of that role,
  `source_profile` of the owner's sign-in, region `eu-west-1`. Expired token:
  ask the owner to sign in again. The AWS names keep the repo's old name on
  purpose (#160).
- Cloud GPU: `aerial-search sagemaker-train`, `ml.g4dn.xlarge`, managed spot
  with checkpoints, EUR 15 per run and EUR 60 per calendar month (owner,
  2026-10-14 #109, confirmed #125). Spot only: AWS declined on-demand
  (#115). Ask `aws-manager` for month-to-date spend first.
- Cloud CPU: `sagemaker-train-head`, `ml.m5.xlarge` spot, 2 h per job, EUR 5
  per month (owner, 2026-10-23, #184). Details and guard: `recipes.md`.
- Laptop (M4, 24 GB): size runs under 30 minutes; a run over one hour goes to
  the cloud within the limits above.

## Documentation

Keep it short, and keep one home for each fact.

| Fact | Home |
|---|---|
| Why the project exists | `INTENT.md` |
| How to work here | this file |
| What is planned or in progress | the board |
| What changed that a reader should know | `CHANGELOG.md`, key items only |
| Why a technical choice was made | `docs/decisions.md`, one line each |
| The evidence behind a decision, when it is long | `docs/<topic>-review.md` |
| What a milestone found | `docs/milestones/<name>.md`, created with the first write-up |
| Raw material for the guide, per closed issue | `docs/guide/notes/<issue>.md`, written at handoff, deleted once in a chapter |
| How the project was built, step by step, with its mistakes | `docs/guide/`, kept current with `/guide` |
| A short, result-led post for a general reader | `docs/blog/`, drafted with `/blog` for the owner to edit |

Write a decision down when a future agent would otherwise have to guess or
re-derive it. Format: `YYYY-MM-DD · chose X over Y · reason`.
When a decision is reversed, leave the old line and prefix it `Superseded by
the <date> line:`; do not delete it.

A rule change in this file is carried into the skill or agent file that
implements it, in the same PR.

### Changelog

`CHANGELOG.md` is for a reader who wants the project's history in two
minutes. Add an entry in the same PR as the change when the change is one of
these:

- a finding that changes what can be trusted, such as a bug in the data
- a result, positive or negative
- a change of direction or method
- a new capability, dataset, or model
- a rule that changes how work is done or merged
- something tried and dropped

Leave out routine fixes, refactors, and wording changes. Group entries by
epic and date, newest first. Each entry is one or two sentences with a link
to its issue or PR. No version numbers.

## Experiments

- **No "works on my machine".** A run whose result is quoted anywhere comes
  from a commit on `main`, with a clean working tree. Merge the code first,
  then run it.
- A run from a branch or a dirty tree is a scratch run. Use it to debug. Never
  quote its numbers.
- Every run is defined by a committed config and a seed, and writes to
  `outputs/<run-name>/` with a `run.json` that records the commit, the config,
  the seed, and the machine.
  `shared/aerial_search/src/aerial_search/run.py` (`start_run`) enforces
  this: it refuses a normal run unless the tree is clean (untracked files
  count) and `HEAD` is on the local `origin/main`, and it fails closed if git
  cannot tell. It never fetches, so run `git fetch` first. Pass `--scratch` to
  `train-ssl`, `train-detector` or `train-head` for a debugging run: it writes to
  `outputs/scratch-<run-name>/` with `"scratch": true` in `run.json`. A run
  directory is never overwritten; pick another `--run-name`.
- The run directory is the only place an experiment writes (there is no
  `--output`). `start_run` inspects the repository that holds the running
  code, not the current directory, and refuses a normal run from a
  non-editable install. Tracked files hidden by skip-worktree or
  assume-unchanged also refuse a normal run. `run.json` records the SHA-256
  of every input (manifests, checkpoint). A normal run given
  `--ssl-checkpoint` needs a `run.json` beside it with `scratch` false,
  `status` completed, and the same `fold` and `view`; the parent's name and
  commit are recorded.
- A normal run is quotable only if its `run.json` has `"status": "completed"`.
  `"started"` means it crashed or is still running; `"failed"` records the
  error. Known limit: files ignored through `.git/info/exclude` are not seen.
- Size runs for the laptop first. Aim for under 30 minutes.
- **Feature cache.** `uv run aerial-search cache-features siglip2-base-naflex
  --camera rgb|thermal [--part K/N] [--collection ID]` runs the frozen
  backbone once per image and writes `outputs/features/<model>-<budget>tok/`,
  read per image or fold with `frozen_head.feature_cache`
  (`read_index`, `read_features`). Each session is a run under the cache's
  `runs/` and goes through `start_run`; the cache's `cache.json` holds the
  model, weight checksum, token budget, pooling and dtype, and a run with any
  of them different is refused. Rerun a crashed session unchanged. Split a
  camera with `--part` to stay under 30 minutes. `cache-features` also caches
  frames only the fold manifests reference (no partner in the other camera);
  `uv run aerial-search check-cache <cache-dir> data/manifests/wisard-full`
  lists any frame the manifests reference that the cache lacks and exits 1. See
  `docs/forward-pass-cost.md`. **The feature cache and the checkpoint
  directory are the two exceptions to "the run directory is the only place an
  experiment writes":** features go to the cache directory, which holds its
  sessions' run directories under `runs/`; `train-baseline --checkpoint-dir`
  (default `checkpoints/` in the run directory) holds the checkpoint, the test
  predictions file and `attempts.json` (#112). Each feature file stays traceable to a commit through its
  `index.jsonl` line (source SHA-256, the writing run's name and commit), that
  run's `run.json` (clean `main` unless scratch, data verification), and
  `cache.json` (weights, revision, settings).
- **Detection head.** `uv run aerial-search train-head rgb|thermal --fold
  <site-day> --percent 1|5|10|100` trains the person-centre head on cached
  features for one camera, fold, fraction and seed, and writes the test
  report to `outputs/detection-head/<cache>/...`. A normal run refuses a
  scratch or incomplete cache. `uv run aerial-search head-table <dir>`
  tabulates completed runs per fold with the mean and spread, and refuses
  scratch or unfinished ones. See `docs/detection-head-design.md`.
- **Cloud run (SageMaker, #113).** For a run too long for the laptop. The
  owner approved up to EUR 15 per run and EUR 60 per calendar month on
  2026-10-14 (#109); the instance is `ml.g4dn.xlarge` on managed spot.
  Spot only: AWS declined the on-demand quota (2026-10-15). Always look
  first:

  ```bash
  uv run aerial-search sagemaker-train rgb --percent 10 --fold 220109_Baker \
    --input tiled --month-to-date-eur 0 --dry-run     # prints the request and cost
  uv run aerial-search sagemaker-train <same flags> --attempt 1   # sends it
  uv run aerial-search sagemaker-fetch <job-name>     # run dir into outputs/
  ```

  It takes `train-baseline`'s flags (device defaults to cuda; no `--data-root`
  or `--checkpoint-dir`: the job sets them). Before sending anything it
  refuses a dirty tree, a HEAD not on `origin/main` (`--scratch` may be off
  main, never dirty), manifests that are not from a normal verified
  `prepare`, and the cost guard: estimate is price x `--expected-hours`
  (default from the laptop figures), worst case is the on-demand price x
  `--max-runtime-hours` (default 12); it refuses a worst case over EUR 15, or
  month-to-date plus this run's worst case plus the worst case of every
  SageMaker job running or stopping, plus the billable time of jobs that
  ended in the last 48 h (Cost Explorer lags; such a job may be counted
  twice, which errs safe), over EUR 60. Month-to-date is read from Cost
  Explorer (account-wide SageMaker spend, up to a day late) or given with
  `--month-to-date-eur`; if unreadable it refuses. Dollars become euros at
  `--eur-per-usd` (default 0.95, high on purpose). The guard estimates; the
  account's budget and the role's conditions enforce. The job's source is
  `git archive` of the commit, with its digest sent along; the job hashes the
  code, manifests and locked dependency list (`uv export`, installed with
  `pip -c`, container torch kept) it received, refuses on a difference, and
  `run.json`
  records the commit, instance type, spot and job name under
  `attestation`, with the data verified in the job as on the laptop. Each
  attempt gets a new run name (`<run>-a<n>-<UTC time>`); the checkpoints live at
  `s3://<bucket>/sagemaker/checkpoints/<run>/`, fixed per run, so a job
  resubmitted with `--attempt 2` resumes. Billable seconds are only known
  afterwards: `sagemaker-fetch` writes them to `cloud-job.json` beside
  `run.json`, after re-archiving the attested commit and checking its digest
  and that it is on `origin/main`. Design and limits:
  `shared/aerial_search/src/aerial_search/cloud/entry.py`.
- **CPU cloud run (#188).** `uv run aerial-search sagemaker-train-head rgb
  --percent 10 --fold 220109_Baker --month-to-date-eur 0 --dry-run` builds one
  `train-head` as a spot job on `ml.m5.xlarge` through the same launcher,
  attestation and guard. The owner approved CPU jobs on 2026-10-23 (#184): at
  most 2 h each (`--max-runtime-hours`, default 2) and EUR 5 a month,
  counted as month-to-date plus running jobs plus this run's worst case, as
  for GPU. Spot is enforced in code only (no IAM condition does). An instance
  not in `cost.INSTANCES` is refused. It takes `train-head`'s flags less
  `--data-root` and `--init-head`; `--eur-per-usd` below 0.95 and a negative
  `--month-to-date-eur` are refused. The EUR 5 counts m5 spend only (Cost
  Explorer grouped by usage type, matching `ml.m5.`; the spelling is
  unverified) and running m5 jobs. The feature cache (#191): the launcher
  checks it as the run will, then sends one camera's features, `cache.json`,
  `index.jsonl` and the writing sessions' `run.json` (about 5.5 GB), once per
  digest, to `sagemaker/inputs/features/<cache>/<digest16>/` with a marker,
  as the manifests. The job hashes the `features` channel against
  `AERIAL_SEARCH_FEATURES`, links it as
  `/opt/ml/feature-links/features/<cache>` and runs `train-head --cache` on
  it; `run.json`'s attestation gains `features_sha256`. Known limit: the
  features digest is verified in the job only; `sagemaker-fetch` cannot
  re-check it, so `run.json`'s job-confirmed `features_sha256` is the
  record. A usage-type format with no `ml.` in it is refused, not read as
  zero. The GPU request is
  pinned by `shared/aerial_search/tests/fixtures/gpu_request.json`.
- Any number quoted in a doc or README must be reproducible from a committed
  config. If a number cannot be traced, remove it.
- Report negative results. State what was expected, what happened, and the
  most likely reason.
- Always compare against a baseline on a held-out split. Splits are by
  site-day (every clip from one site on one date), never by clip or frame,
  because neighbouring frames are near-duplicates and clips from one site-day
  share terrain and people. Each labelled site-day is the test set of one
  fold. Report every result per fold and as the mean and spread over the
  folds, never as one pooled number. See `docs/site-folds-review.md`.
- Every published table or figure that shows `ap_iou25` shows `ap_iou50`
  beside it. The lower threshold is the primary metric for tiny people, and
  the reader must be able to see how much it contributes. See
  `docs/evaluation-metric-review.md`.

## Tests

Test what would silently corrupt results: pairing, splits, label handling,
metrics. Skip tests that only restate the implementation.
