---
name: tidy
description: Audit the repository for drift and clutter, then apply the safe clean-ups on a branch. Use at the end of a milestone, or when docs, code, and board have drifted apart.
---

# Tidy

## Steps

1. Delegate the audit to the `repo-janitor` agent. It is read-only and returns
   a report grouped by check.
2. Sort its findings into two groups:
   - **Do:** lint fixes, broken links, stale paths in docs, unused imports,
     dead code, superseded docs, unused dependencies, merged branches.
   - **Needs owner:** anything under "Ask the owner first" in `CLAUDE.md`,
     such as deleting data or a remote branch with unmerged work.
3. Apply the first group. Put the second group to the owner as direct
   questions.
4. Make the changes on a branch named `chore/tidy-<date>`, one logical change
   per commit.
5. Run `uv run pytest` and `uv run ruff check .`.
6. Review before merge as `CLAUDE.md` describes, and say in the PR which
   review happened.
7. Open a PR as the `flow-github` playbook describes. Tell the owner what changed and what
   was left.

## Rules

- Tidy does not add features or change behaviour.
- A number in a doc that cannot be traced to a run is removed, not corrected
  by guesswork.
- Delete superseded files. Do not move them to an archive folder.
