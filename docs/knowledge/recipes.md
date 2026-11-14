# Recipes

How to run the things this repo does often. Procedures only; project facts are
in `CLAUDE.md`, traps in `gotchas.md`.

## Set up and check

```bash
uv sync --dev                  # install the workspace: shared and every project
uv run pytest                  # tests of shared/ and projects/
uv run pytest projects/<name>  # one project's tests
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run ty check                # types
tools/install_archify.sh       # one-time: install the Archify diagram skill
```

- CI runs all four checks: ruff check, ruff format, ty, and pytest.
- The root is a uv workspace, not a package. Its `dev` group lists every
  member, so `uv sync --dev` installs them all without `--all-packages` (the
  org CI runs plain `uv sync --dev --locked`).
- Before a timed laptop run, check the laptop is on mains power and run it
  under `caffeinate -dims`, so sleep or a locked screen cannot spoil it.

## Work the board from the command line

```bash
# attach issue <child> to epic <parent> (needs the child's id, not its number)
CHILD_ID=$(gh api repos/graylayer-labs/aerial-search-lab/issues/<child> --jq .id)
gh api -X POST repos/graylayer-labs/aerial-search-lab/issues/<parent>/sub_issues \
  -F sub_issue_id=${CHILD_ID}

gh project item-list 1 --owner graylayer-labs --format json --limit 100   # status
# to read one field, add --jq to this command (see gotchas.md)

# read a task in full
gh issue view <n> --json title,body,labels,comments --jq \
  '.title, ([.labels[].name] | join(", ")), .body,
   (.comments[] | "--- comment " + .createdAt, .body)'

# move an item (get <item-id> from item-list; option ids are in CLAUDE.md)
gh project item-edit --project-id PVT_kwDOEDMalc4BlE5m --id <item-id> \
  --field-id PVTSSF_lADOEDMalc4BlE5mzhjzG-Q --single-select-option-id <option>
```

## Add an approach

Start from `projects/_template` (steps in `projects/README.md`). Write its
expected results in its README before its first run. It shares the
evaluation (`aerial_search`) and changes none of it; register its commands in
`PROJECT_COMMANDS` in the shared CLI so they run as `aerial-search <command>`,
in a cloud job too.

## Make a finished diagram

- **Mermaid** for drafts and working documents (cheap, GitHub renders it).
- **Archify** for finished figures: the README, milestone write-ups, blog
  posts. Figures and their source JSON live in `docs/figures/`, for example
  `pipeline.svg` and `pipeline.archify.json`.
- Install once with `tools/install_archify.sh`. It fetches a pinned, verified
  commit into `.claude/skills/archify/`, which is git-ignored, and refuses to
  install if the commit differs. `--force` replaces an existing install.
- Limits, enforced in `.claude/settings.json`: no brand URLs in a diagram
  (brand capture fetches from the network), never pass `--open`, never run
  `preview`.
- Archify has no command-line SVG export. Export from the viewer's Export
  menu, and commit only the SVG and its JSON, never anything from `.archify/`.
  Check the SVG for absolute paths first.

## Commit signing setup

Every commit is signed, on branches as well as on `main`. The repository's
local config signs with an SSH key made for this project,
`~/.ssh/ssl_aerial_signing_ed25519`, registered on the owner's GitHub account
as a signing key. The key has no passphrase so that agents can sign; it signs
commits and nothing else.

- Agent worktrees share the repository's config, so they sign too.
- If a commit fails with a signing error, the key is missing: a fresh machine
  needs its own key, registered on GitHub, and the three `git config --local`
  lines in issue #27.
- Never copy the private key anywhere, and never commit it.

## Pin, check and fetch the data

The dataset is pinned by
`shared/aerial_search/src/aerial_search/data/checksums/wisard-full.tsv.gz`: one line per file (path, size, SHA-256), 100,794 files, 3.9 MB. Its root hash
(SHA-256 of the list's text) is
`6e5e5d554c4be119f15bb636049800788904f3a2e2759cabce402c676d1575ed`.
Hashing all 43 GB takes about 25 seconds.

- A normal `prepare` hashes the directories of its selected collections and
  checks them against the list before writing anything; on a difference it
  refuses and names the files. `data_quality.json` records
  `data: {dataset, root_hash, verified, directories, seconds}` and
  `provenance: {commit, scratch}`. A source that is not the pinned dataset
  records `verified: false` and no root hash. `prepare --scratch` skips the
  check and records `verified: false`; it will not overwrite manifests that
  are not scratch unless `--force`.
- A normal run (`train-ssl`, `train-detector`, `train-head`) refuses manifests whose
  `data_quality.json` is missing, scratch, or unverified, re-verifies the
  directories those manifests reference against the list before starting,
  and copies `data` and `provenance` into `run.json` with `data_verified_at`.
  A scratch run skips the check and records `data_verification_skipped`.
  So a quotable result's `run.json` names the data it was verified against.

```bash
uv run aerial-search check-data data/raw/wisard-full   # exit 1 and list every changed/missing/added file
uv run aerial-search check-data data/raw/wisard-full --directory 220109_Baker_Enterprise_IR_2
uv run aerial-search fetch-directory <name> --data-root data/raw/wisard-full   # from S3 if absent, then verified
uv run aerial-search checksum data/raw/wisard-full --force   # re-pin; only if the dataset deliberately changes
```

- `fetch-directory` copies into a temporary `.fetch-<name>-*` directory and
  moves it into place only if it matches the list; a mismatch deletes the
  copy. A directory already present that differs is refused and kept. The
  AWS profile is `AWS_PROFILE` from the environment, else from
  `.claude/settings.json`, else none (default credentials); the log line
  says which. A failed `aws` call prints its error text. The function is
  `aerial_search.data.checksums.ensure_directory`.
- `prepare` records the commit and refuses a dirty tree or a commit off
  `origin/main`; pass `--scratch` for a debugging run.

## Regenerate manifests, folds and pairing sheets

Which RGB frame goes with which thermal frame is decided in
`shared/aerial_search/src/aerial_search/data/wisard.py`; the evidence is in
`docs/wisard-pairing-review.md`. Only directory pairs listed in
`WISARD_COLLECTIONS` are paired. Pairs are 0 to 2 frames apart in time; the
cameras drift.

```bash
uv run aerial-search prepare data/raw/wisard-full --output data/manifests/wisard-full
uv run aerial-search folds data/raw/wisard-full data/manifests/wisard-full
uv run aerial-search check-folds data/raw/wisard-full data/manifests/wisard-full
uv run python tools/pairing_contact_sheets.py data/raw/wisard-full \
  data/manifests/wisard-full/all_pairs.jsonl outputs/pairing-check
```

## Run an experiment

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

## Run on the cloud GPU (SageMaker, #113)

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

## Run a CPU cloud job (#188)

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
