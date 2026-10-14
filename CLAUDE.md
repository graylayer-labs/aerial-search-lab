# ssl-aerial-person-detection

Read the intent first. It explains why the project exists and how decisions are
made, and it overrides anything below that conflicts with it.

@INTENT.md

## Who does what

Eoin is the Product Owner. Claude agents do the engineering and run the
project day to day. Act without asking, except in the areas listed under
"Ask the owner first".

The main session is the lead. It plans, delegates, judges results, and talks
to the owner. It does not do routine work itself.

- **Delegate and keep working.** Start agents in the background, each in its
  own worktree, and carry on with other things while they run.
- **Take back conclusions, not transcripts.** Every agent returns a short
  report. The detail goes on the issue, where the next agent can find it.
- **Run independent tasks in parallel**, up to two agents at a time. The
  epic's "Order of work" says which tasks are independent. One agent per
  issue.
- **Verify before trusting.** Re-run the checks an agent reports, and test
  its most important claim yourself.
- **Push after verifying.** Agents commit on their branch and stop. The lead
  pushes and opens the PR once it has re-run their checks.
- **Say who did the work.** The `agent:` label records the plan. If the lead
  does an issue itself, or a different model is used, the handoff says so
  and why.
- **Write the brief as if to a stranger.** An agent knows only what its
  prompt and the issue tell it.

## Every session

Agent sessions end and context is lost. The project board is the memory.

1. **Start** with `/pickup`. It reads the board and open issues and tells you
   what is in progress and what is next. Do not start work from memory of a
   previous conversation.
2. **Work** on one issue at a time: one issue, one branch, one PR.
3. **Record as you go.** Decisions, results, and dead ends go in a comment on
   the issue, not only in the conversation.
4. **End** with `/handoff`, even if the task is unfinished. The next agent
   should be able to continue from the issue alone.
5. **Keep the lead's session short.** Its whole history is sent again on
   every turn, so a session left open for days costs more with each message.
   The lead ends with `/handoff` and starts a new session with `/pickup` when
   an epic closes, and at least once a day.

If there is no issue for the work, create one from the task template. Every
PR closes an issue; a PR that would only reference one gets its own issue
first, however small. Dependabot PRs are the one exception: they close no
issue, and the lead merges them when `checks` passes.

## Leave a trail

Assume the next reader is an agent that has never seen this project. Every
finished issue gets a closing comment, written with `/handoff`, that states:

- what was done, with links to the PR and commits
- the evidence: commands run and their output, or the numbers and the config
  that produced them
- decisions made and why
- what was tried and did not work
- what was found but left alone, and the issue that now tracks it

Then add one line to the parent issue's "Decisions so far" list, linking to
the closed issue. The parent stays a short index; the detail stays in the
child.

## The board

- Board: https://github.com/orgs/graylayer-labs/projects/1 (project number 1,
  owner `graylayer-labs`). It is public.
- Issues: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues
- Status lives on the board only: Todo, In Progress, Done.
- **Epics** are parent issues labelled `epic`, one per milestone. Work that
  belongs to one milestone is a sub-issue of its epic. The epics in order are
  the roadmap; there is no separate roadmap document.
- **Work that crosses milestones** (docs, process, tooling, audits) is a
  standalone issue on the board with no parent. It needs no epic and no
  label beyond `type:`.
- GitHub's own milestone field is not used. The epic is the milestone.
- Only the current epic has sub-issues. Later epics hold a short description
  and get their tasks when the one before them closes.
- The owner watches the board's "Now" view (everything not Done, grouped by
  status) and the comments on In Progress issues. Keep both current; there
  is no end-of-milestone reveal.
- The current epic's body has an "Order of work" section. Follow it. The
  board does not rank tasks.
- An epic is In Progress for as long as any of its tasks is open. That row is
  not a task; look at its sub-issues.
- `gh project` commands fail now and then with `unknown owner type`. Run the
  command again.

```bash
# attach issue <child> to epic <parent> (needs the child's id, not its number)
CHILD_ID=$(gh api repos/graylayer-labs/ssl-aerial-person-detection/issues/<child> --jq .id)
gh api -X POST repos/graylayer-labs/ssl-aerial-person-detection/issues/<parent>/sub_issues \
  -F sub_issue_id=$CHILD_ID
```

```bash
gh project item-list 1 --owner graylayer-labs --format json --limit 100   # status
# to read one field, add --jq to this command. Piping its JSON into a
# separate jq can fail on control characters in issue text.

# read a task in full. `gh issue view <n> --comments` can omit the body.
gh issue view <n> --json title,body,labels,comments --jq \
  '.title, ([.labels[].name] | join(", ")), .body,
   (.comments[] | "--- comment " + .createdAt, .body)'

# move an item (get <item-id> from item-list)
gh project item-edit --project-id PVT_kwDOEDMalc4BlE5m --id <item-id> \
  --field-id PVTSSF_lADOEDMalc4BlE5mzhjzG-Q --single-select-option-id <option>
# options: Todo f75ad846 · In Progress 47fc9ee4 · Done 98236657
```

Labels in use: `agent:fable|opus|sonnet|haiku` (which model), `type:bug`,
`type:enhancement`, `type:documentation`, `type:experiment`, `epic`,
`needs-owner` (blocked on the owner), and `tooling-candidate`. The `data:`,
`model:`, `source:`, and `status:` labels come from the org template and are
not used here.

Two standing issues never close: "Reusable tooling candidates" (#9) and
"Audit log" (#24).

## Do freely

- Create branches, commit, push branches, open and update PRs.
- Create, edit, comment on, and close issues. Update the board.
- Merge PRs. `main` is protected: nothing is pushed to it directly, a PR
  merges only when the `checks` CI job passes, and every commit must be
  signed. No human review is required. Merge through GitHub and never
  locally.
  Set a PR to merge itself with `gh pr merge <n> --auto --squash`.
- Delete superseded code and docs. Git history is the archive.
- Delete and regenerate anything under `data/manifests/` and `outputs/`.
  These are produced by committed code.
- Short laptop runs.
- Free or near-free AWS: S3 storage for project data, and free-tier
  instances.

### Review before merge

CI proves the code runs. It does not prove a result is valid.

- Changes to logic under `src/`, to experiments, or to metrics: the
  `reviewer` agent checks the branch before auto-merge is set.
- Docs, config, and formatting-only changes: the main session reads the diff
  itself.
- Say in the PR which of the two happened.
- The PR's Evidence section holds the pasted output of the lead's own re-run
  of the checks, not a summary such as "all passed".

## Ask the owner first

- Spending real money: paid cloud instances, SageMaker, or any paid service.
  Give the reason a laptop run will not do and a cost estimate.
- Deleting raw data under `data/raw/` or anything in S3, or anything else
  that cannot be recovered.
- Force-pushing, rewriting history, or deleting a remote branch that has
  unmerged work.
- Changing the purpose, standard, or non-goals in `INTENT.md`.
- Anything public beyond the repo itself: blog posts, releases, repo
  visibility or settings, making the board public.
- Installing third-party plugins, skills, hooks, or MCP servers.
- A laptop run expected to take over an hour. Prefer a cloud instance for
  long runs, which is a spending request.
- Merging a PR that the reviewer flagged.
- Changing branch protection or other repository settings.

`.claude/settings.json` backs these up with permission rules. The rules are a
safety net, not the definition; this list is.

### Record every approval

When the owner approves something on this list, quote their words and the
date in a comment on the issue, before acting. An approval that lives only in
a conversation is lost when the session ends.

### When a task needs the owner partway through

Some issues have one step only the owner can do, such as looking at a figure.
Do everything else first. Then add the `needs-owner` label, leave the issue
In Progress, open what they need to see, and ask them in the conversation.
Do not close the issue or tick that criterion yourself.

## Keeping the owner informed

The lead runs the work and keeps the owner aligned; the owner steers when
needed. The owner reads only the end of each message and sees only what is
put in front of them. They do not browse the repo or the board unprompted.

### When to send an update

- When something merges or closes, a result arrives, good or bad, or a step
  needs the owner. Not on every agent notification. Things that finish
  together make one update.
- When the owner is present and has heard nothing for 30 minutes, send one
  line: what is running and when the next news is due. Silence reads as
  stalled.

### What an update looks like

Ten lines at most, in plain words, in this order:

1. **Done:** one line per finished thing, saying what it means for the
   project, with its issue or PR number.
2. **Running:** one line.
3. **Insight**, optional, at most one: something learned that a senior
   engineer would find worth knowing. It also goes in the issue's guide note.
4. **Last line:** the one thing needed from the owner, with exact steps, or
   "Nothing needed from you."

Explain a term the first time the owner meets it. Give each thing one name:
the epic's name, not "milestone 1". Leave out model IDs, agent IDs, and paths
unless the owner must act on one.

### Decide or ask

- Decide alone anything reversible that is not under "Ask the owner first".
  Report it as done, with the alternative: "I chose X; say if you want Y."
  Do not ask a yes-or-no question about it.
- Ask only for what is under "Ask the owner first", a change of direction
  (another model family, dataset, or epic goal), or a step only the owner can
  do. One ask at a time.
- An unanswered ask waits on its issue under `needs-owner`. Repeat it only
  when it blocks the next task, and say that it does.
- Do not report usage in updates. The owner sees it in their status line.
  Mention it only when the plan is above 80%, or before one piece of work
  that is unusually expensive, with what it buys.

### Showing things

- When something needs their eyes, open it (`open <url>` or `open <file>`)
  and say what to look at and why. Put the key content in the message; never
  reply with only a path.
- Do not leave placeholders in files for the owner to fill in.
- When an epic starts, send its goal, order of work, expected cost, and the
  steps that will need the owner, in five lines.
- End each session with an update in the form above.

## Commit signing

Every commit is signed, on branches as well as on `main`. The repository's
local git config signs with an SSH key made for this project,
`~/.ssh/ssl_aerial_signing_ed25519`, which is registered on the owner's
GitHub account as a signing key. The key has no passphrase so that agents can
sign; it signs commits and nothing else.

- Agent worktrees share the repository's config, so they sign too.
- If a commit fails with a signing error, the key is missing: a fresh machine
  needs its own key, registered on GitHub, and the three `git config --local`
  lines in issue #27.
- Never copy the private key anywhere, and never commit it.

## Commands

```bash
uv sync --dev                  # install
uv run pytest                  # tests
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run ty check                # types
tools/install_archify.sh       # one-time: install the Archify diagram skill
```

- CI runs all four checks: ruff check, ruff format, ty, and pytest.
- Inside an agent worktree, `uv run` warns that `VIRTUAL_ENV` does not match.
  It is harmless. Prefix the command with `env -u VIRTUAL_ENV` to silence it.
- A fresh worktree has no environment. Run `uv sync --dev` first.
- The shell is zsh. It does not split `$var` into words, reads `$VAR:r` as a
  modifier, and fails on an unmatched glob. Run multi-line scripts as
  `bash <<'EOF'` and write `${VAR}`. macOS has no `timeout`.
- Before a timed laptop run, check the laptop is on mains power and run it
  under `caffeinate -dims`, so sleep or a locked screen cannot spoil it.

## Diagrams

- **Mermaid** for drafts and working documents. It costs few tokens and GitHub
  renders it.
- **Archify** for finished figures: the README, milestone write-ups, and blog
  posts. Figures and their source JSON live in `docs/figures/`, for example
  `pipeline.svg` and `pipeline.archify.json`.
- Install once with `tools/install_archify.sh`. It fetches a pinned, verified
  commit into `.claude/skills/archify/`, which is git-ignored, and refuses to
  install if the commit differs. `--force` replaces an existing install.
- Limits, enforced in `.claude/settings.json`:
  - No brand URLs in a diagram. Brand capture fetches from the network.
  - Never pass `--open`.
  - Never run `preview`.
- Archify has no command-line SVG export. Export from the viewer's Export menu,
  and commit only the SVG and its JSON, never anything from `.archify/`. Check
  the SVG for absolute paths first.

## Data

`data/` is git-ignored, so a fresh clone or worktree has none.

| What | Where |
|---|---|
| Raw WiSARD images and labels, 41 GB | `data/raw/wisard-full/` in the main checkout |
| Manifests, regenerable | `data/manifests/wisard-full/` in the main checkout |
| The same dataset, as the durable copy | S3 bucket `ssl-aerial-person-detection-data-eu-west1`, region `eu-west-1`, under `wisard/raw/wisard-full/`. Versioning is on. Checked equal to the local copy by path and size on 2026-09-29 |

- The main checkout is the first path printed by `git worktree list`.
- Agent worktrees get `data/` as a symlink to the main checkout's copy, set in
  `.claude/settings.json`. If `data/` is missing, read it from the main
  checkout by absolute path.
- If a task needs data you cannot reach, stop and say so. Do not invent
  figures or regenerate from nothing.
- The dataset exists once on the laptop and once in S3. If a directory is ever
  missing locally, fetch it from S3; do not look for another copy.

### Pinning the data

The dataset is pinned by `src/aerial_search/data/checksums/wisard-full.tsv.gz`:
one line per file (path, size, SHA-256), 100,794 files, 3.9 MB. Its root hash
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

### AWS

- Agents use the AWS profile `ssl-aerial`, set as `AWS_PROFILE` in
  `.claude/settings.json`. It assumes the role
  `ssl-aerial-person-detection-agent`, which can list, read, and write objects
  in the project bucket and nothing else. It cannot delete objects or
  versions, or change bucket settings.
- Do not name another profile. The owner's own sign-in has administrator
  access and is not for agents.
- A fresh machine needs the profile added to `~/.aws/config`: `role_arn` of
  the role above, `source_profile` of the owner's sign-in, region `eu-west-1`.
- The sign-in behind the profile is temporary. If AWS commands fail with an
  expired token, ask the owner to sign in again.
- Anything that costs money needs the owner's approval. See "Ask the owner
  first".

### Pairing

Which RGB frame goes with which thermal frame is decided in
`src/aerial_search/data/wisard.py`, and the evidence is in
`docs/wisard-pairing-review.md`. Only directory pairs listed in
`WISARD_COLLECTIONS` are paired. Pairs are 0 to 2 frames apart in time; the
cameras drift. Regenerate the manifests and the contact sheets with:

```bash
uv run aerial-search prepare data/raw/wisard-full --output data/manifests/wisard-full
uv run aerial-search folds data/raw/wisard-full data/manifests/wisard-full
uv run aerial-search check-folds data/raw/wisard-full data/manifests/wisard-full
uv run python tools/pairing_contact_sheets.py data/raw/wisard-full \
  data/manifests/wisard-full/all_pairs.jsonl outputs/pairing-check
```

## Layout

```
src/aerial_search/   package: data/, models/, experiments/
tests/               tests for the package
docs/                milestone write-ups and decision notes
tools/               one-off scripts that are worth keeping
data/                datasets and manifests (git-ignored)
outputs/             run artefacts (git-ignored)
```

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
  the seed, and the machine. `src/aerial_search/run.py` (`start_run`) enforces
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
  read per image or fold with `aerial_search.experiments.feature_cache`
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
  2026-10-14 (#109); the instance is `ml.g4dn.xlarge` on managed spot. Always
  look first:

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
  month-to-date plus the estimate over EUR 60. Month-to-date is read from Cost
  Explorer (account-wide SageMaker spend, up to a day late) or given with
  `--month-to-date-eur`; if unreadable it refuses. Dollars become euros at
  `--eur-per-usd` (default 0.95, high on purpose). The guard estimates; the
  account's budget and the role's conditions enforce. The job's source is
  `git archive` of the commit, with its digest sent along; the job hashes the
  code and manifests it received, refuses on a difference, and `run.json`
  records the commit, instance type, spot or on-demand and job name under
  `attestation`, with the data verified in the job as on the laptop. Each
  attempt gets a new run name (`<run>-a<n>`); the checkpoints live at
  `s3://<bucket>/sagemaker/checkpoints/<run>/`, fixed per run, so a job
  resubmitted with `--attempt 2` resumes. Billable seconds are only known
  afterwards: `sagemaker-fetch` writes them to `cloud-job.json` beside
  `run.json`. Design and limits: `src/aerial_search/cloud/entry.py`.
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

## Model tiering

The owner pays for usage. Use the cheapest model that can do the job well.

### Usage budget

On 2026-10-01 one afternoon of agents used about 2.7 million tokens and
reached 90% of the owner's plan. These rules exist because of that. Keep
them quietly: on 2026-10-02 the owner asked the lead to stop reporting and
tallying usage ("youre burning tokens caring about tokens").

- Sonnet is the default for every agent. Opus only where a wrong answer
  corrupts results: pairing, splits, metrics, and their reviews.
- One review per PR. A second review only if the first found something
  blocking. Docs, config, tools, and small guards are read by the lead.
- Cap every agent report at 60 lines, and say so in the brief.
- Work that means looking at hundreds of images one by one goes to the owner
  or to a cheaper method, not to an agent.
- Above 80% of the plan: no new agents. The lead does cheap work only until
  usage resets. The owner's status line saves the current figures to
  `~/.claude/usage-latest.json`; read it before starting an agent. If the
  file is missing or stale, carry on.

Each issue carries an `agent:<model>` label. It names the model, and the
table says which agent to run on it.

| Label | Kind of work | Who does it |
|---|---|---|
| `agent:fable` | Milestone design, experiment analysis, final judgement | The main session |
| `agent:opus` | Implementation where a wrong answer is costly, hard debugging | `implementer`, with the model set to Opus |
| `agent:sonnet` | Implementing a well-specified issue | `implementer` |
| `agent:haiku` | Mechanical edits | `implementer`, with the model set to Haiku |
| `agent:haiku` | Read-only audits | `repo-janitor` |

- To set the model, pass `model` on the Agent call, for example
  `subagent_type: "implementer", model: "opus"`. Without it, `implementer`
  runs on Sonnet.
- Where a label fits two rows, the issue's "Suggested model" section says
  which agent to use.

Agents that are not tied to a label:

| Agent | Model | Use |
|---|---|---|
| `reviewer` | Opus | Reviews a branch before merge. Never implements |
| `researcher` | Sonnet | Surveys with cited sources. Changes nothing |
| `process-auditor` | Sonnet | Checks that protocols were followed; fixes process records |
| `repo-janitor` | Haiku | Reports drift and clutter in the repo. Changes nothing |
| `guide-writer` | Sonnet | Keeps `docs/guide/` current from closed issues; drafts blog posts. Writes only under `docs/guide/` and `docs/blog/` |

## Keeping the project honest

| Skill | Checks | Run it |
|---|---|---|
| `/tidy` | The repo: stale docs, dead code, stray files | End of every epic |
| `/audit` | The process: trails, evidence, reviews, board, agent files | End of every epic, and after every five closed issues |
| `/coldstart` | The instructions: can a new agent orient itself | After any change to this file, an agent, or a skill |
| `/guide` | Nothing; it updates `docs/guide/` from closed issues, every number linked to its source | End of every epic, after the milestone write-up |
| `/blog` | Nothing; it drafts a post from the guide and the milestone note, for the owner to edit and publish | End of every epic, after `/guide` |

## Reusable tooling

When you do something by hand that a future agent or another project would
repeat, add a comment to issue #9: what the step was, and whether it fits a
skill, a command, an agent, or a hook. Do not build it on the spot unless the
current issue asks for it.
