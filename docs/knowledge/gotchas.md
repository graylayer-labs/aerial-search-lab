# Gotchas

Traps in this repo, and things that look wrong but are right.

## zsh does not split `$var` and fails on unmatched globs

- Cause: the shell is zsh. It does not word-split `$var`, reads `$VAR:r` as
  a modifier, and errors on a glob with no match.
- Instead: run multi-line scripts as `bash <<'EOF'` and write `${VAR}`.

## `timeout` is not found

- Cause: macOS has no `timeout`.
- Instead: bound a run another way (the tool's own limit, or `perl -e alarm`).

## `gh project` fails with `unknown owner type`

- Cause: an intermittent `gh` error.
- Instead: run the command again.

## `gh issue view <n> --comments` shows no body

- Cause: it can omit the body.
- Instead: `gh issue view <n> --json title,body,labels,comments --jq '...'`
  (command in `recipes.md`).

## Piping `gh project item-list` JSON into jq fails

- Cause: control characters in issue text.
- Instead: add `--jq` to the `gh` command itself.

## `uv run` warns that `VIRTUAL_ENV` does not match

- Cause: agent worktrees run under another environment. Harmless.
- Instead: prefix with `env -u VIRTUAL_ENV` to silence it.

## A fresh worktree has no environment

- Instead: `uv sync --dev` first.

## `data/` is missing in a worktree

- Cause: `data/` is git-ignored. Agent worktrees get it as a symlink to the
  main checkout's copy (set in `.claude/settings.json`).
- Instead: read it from the main checkout by absolute path (the first path
  printed by `git worktree list`). Never regenerate from nothing.

## CI Python version drifted from 3.12 to 3.14

- Cause: the org's shared CI moved its default Python.
- Evidence: #215.

## AWS names say `ssl-aerial-person-detection`, not `aerial-search-lab`

- Cause: on purpose. Renaming the bucket, roles and `project` tag would mean
  new resources and a data copy for no gain.
- Evidence: #160.
