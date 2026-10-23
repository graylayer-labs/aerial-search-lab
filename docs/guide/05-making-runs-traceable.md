# 5. Making runs and records traceable

*Last updated 2026-10-01 from issues [#20], [#22], [#23], [#28], [#29], [#30]
and [#38]. Issue #38 has no closing handoff comment yet, so its part of this
chapter rests on `CLAUDE.md`, the changelog and its issue text.*

## What we set out to do

The owner's rule is "no works on my machine". A number quoted anywhere must
come from a commit on `main`, with a clean working tree, and must say which
code and which data produced it ([#23]). The same goes for the work around the
code. Agents lose their memory when a session ends, so what was done and why
has to be written where the next agent can find it ([#10 handoff][i10-handoff]).

## What we found

**The run module passed its criteria and failed its review three ways.**
`shared/aerial_search/src/aerial_search/run.py` has `start_run` and `finish_run`. A normal run is
refused unless the tree is clean and `HEAD` is on `origin/main`. A
`--scratch` run is allowed from anywhere, written to `outputs/scratch-<name>/`
and marked `"scratch": true` in `run.json`. The first version did all of that.
A reviewer then showed that a result could still look quotable in three ways
([#23 handoff][i23-handoff]):

1. Results could be written outside the labelled run directory.
2. The wrong repository could be inspected. With agents in worktrees, the
   current directory and the code's directory often differ.
3. Inputs had no provenance.

The fixes: the run directory is the only place an experiment writes, so
`--output` was removed. The repository is found from the location of the
running code. `run.json` records the SHA-256 of every input, and a normal run
may start only from a checkpoint made by a normal, completed run. The lead
added that last check itself, because it was a few lines ([#23 handoff][i23-handoff]).
44 tests passed at the end ([#23 handoff][i23-handoff]).

**Folds are part of the trail.** A run on a fold records `fold` and `view` in
`run.json`. A normal run refuses a checkpoint from another fold or view, since
pretraining in the MtErie fold has seen Baker ([folds review][folds-review],
[#3 handoff][i3-handoff]).

**A run could not say which data it used, until #38.** The run module
recorded a hash of each manifest and nothing about the images they point to
([#38]). The dataset exists once on the laptop and once in S3, with versioning
on, and an AWS role that can write but never delete ([#7 actions][i7-actions]).
PR [#47] pinned it: a committed list of 100,794 files with size and SHA-256
(3.9 MB gzipped, root hash `6e5e5d55...75ed`). A normal `prepare` checks the
directories it reads against the list and refuses on any difference. A normal
run re-checks the directories its manifests reference, and `run.json` records
what was verified. `prepare` records its own commit and refuses a dirty tree or
a commit off `origin/main`. `aerial-search check-data` names every changed,
missing or added file ([changelog](../../CHANGELOG.md), [CLAUDE.md](../../CLAUDE.md)).
It chose a per-file list over per-directory hashes, because a check must name
every file that differs ([decisions](../decisions.md)).

**The working record needed the same care.** What was found, and what we did:

- A cold-start test, a fresh agent with no context reporting what confused it,
  found 9 gaps on its first run ([#20]). A second run did not repeat findings 1
  to 8 and reported 7 smaller ones, 6 of which were fixed ([#20 handoff][i20-handoff]).
  The test is now the `/coldstart` skill.
- Requiring signed commits on `main` was switched on and off. PR #25 sat
  blocked with CI green, because the rule also demands signatures on branch
  commits ([#22 handoff][i22-handoff]). Branch commits are now signed with a
  project key; see "Commit signing" in `CLAUDE.md` and issue #27.
- The first process audit had 13 findings ([epic #13][epic]). Evidence had been
  summarised and not pasted in five PRs and three handoffs, and owner approvals
  lived only in conversation ([#28]). The PR template and the `implementer`
  report now carry an example of pasted output, and `CLAUDE.md` requires
  approvals to be quoted on the issue ([#28 handoff][i28-handoff]).
- `CHANGELOG.md` records key changes only, newest first ([#29]).
- Finished diagrams use Archify, installed by a script that refuses any commit
  but one pinned, because its 7.3 MB of code stays out of the repository
  ([#30 handoff][i30-handoff], [decisions](../decisions.md)).

## Decisions and why

- **An existing run directory is refused, not suffixed.** Two runs can never
  share a name ([#23 handoff][i23-handoff]).
- **Nothing is fetched from the network.** Run `git fetch` before a run
  ([#23 handoff][i23-handoff]).
- **Only runs with `"status": "completed"` are quotable.** `started` means
  crashed or still running ([CLAUDE.md](../../CLAUDE.md)).
- **The AWS role may write but not delete,** with versioning on, so an
  overwrite can be undone ([#7 actions][i7-actions]).
- **A separate process auditor,** which checks how work was done, over widening
  the repo janitor ([decisions](../decisions.md)).
- **Evidence is pasted, and approvals are quoted on the issue** ([#28]).

## How to reproduce

You cannot run a real experiment yet. No training run had gone through the
module when [#23] closed ([#23 handoff][i23-handoff]). You can watch it
refuse. On 2026-10-01,
from a branch with a dirty tree, then from a clean tree not on `main`:

```bash
uv run aerial-search train-ssl --fold 220109_Baker --run-name guide-check \
  --data-root data/raw/wisard-full --manifests data/manifests/wisard-full
```

```text
refusing to start: the working tree has uncommitted changes (including untracked files not in .gitignore). Commit them, or pass --scratch:
refusing to start: HEAD <sha> is not on origin/main. Merge it first, or pass --scratch. If it was merged recently, a `git fetch` may be needed.
```

To check the data against the pinned list:

```bash
uv run aerial-search check-data data/raw/wisard-full
```

On 2026-10-01 it printed `OK   data/raw/wisard-full matches ... (root hash
6e5e5d554c4be119f15bb636049800788904f3a2e2759cabce402c676d1575ed)` in about 20
seconds. `CLAUDE.md` says hashing all of it takes about 25.

The first message is shown without the list of changed files that follows it.
Both runs wrote nothing under `outputs/`. We pointed `--manifests` at ones made
with `prepare --scratch`, which a normal run would also refuse.
The tests that pin each rule are in `shared/aerial_search/tests/test_run.py`:

```bash
uv run pytest shared/aerial_search/tests/test_run.py
```

## What we would do differently

- **State the property in the acceptance criteria.** The criteria described the
  module, not the guarantee it had to give, so a result could look quotable
  and not be. "A result is quotable only if its commit, config and inputs are
  recorded" would have caught all three ([#23 handoff][i23-handoff]).
- **Check the instructions as you check code.** Cold-start tests and audits
  found gaps the authors could not see ([#20 handoff][i20-handoff]).
- **Known limits** when [#23] closed ([#23 handoff][i23-handoff]): files
  ignored through `.git/info/exclude` are not seen by the clean-tree check,
  and the command line imports `torch` before it refuses. Raw images were not
  hashed then; since [#47] they are checked against the pinned list.

[#20]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/20
[#22]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/22
[#23]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/23
[#28]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/28
[#29]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/29
[#30]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/30
[#47]: https://github.com/graylayer-labs/ssl-aerial-person-detection/pull/47
[#38]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/38
[epic]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/13
[i3-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/3#issuecomment-5936845727
[i7-actions]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/7#issuecomment-5893642919
[i10-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/10#issuecomment-5891129310
[i20-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/20#issuecomment-5891603601
[i22-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/22#issuecomment-5891604000
[i23-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/23#issuecomment-5892333572
[i28-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/28#issuecomment-5891760114
[i30-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/30#issuecomment-5892823052
[folds-review]: ../site-folds-review.md
