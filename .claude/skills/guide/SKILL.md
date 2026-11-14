---
name: guide
description: Update the follow-along guide in docs/guide/ from the issues closed since it was last updated. Use after a milestone write-up, or whenever several issues have closed.
---

# Guide

## Steps

1. Run `git fetch` and start from `main`.
2. Delegate to the `guide-writer` agent. It finds the new issues itself: closed
   issues that no chapter in `docs/guide/` names in its "Last updated" line,
   and that the README does not list as not yet covered. Tell it the date and
   anything the owner asked to be added. Do not tell it what to write.
3. Read its report. Open one chapter it changed and check it against its
   sources: pick five numbers and trace each to the issue or note it links.
   If any cannot be found, send the chapter back.
4. Run the commands in the chapter that changed, or confirm the agent did, on a
   scratch path.
5. Run `uv run ruff check .` and `uv run ruff format --check .`.
6. This is a docs-only change, so the main session reads the diff itself, as
   `CLAUDE.md` (Repo rules) describes. Say so in the PR.
7. Push and open the PR once the checks pass. Tell the owner which chapters
   changed and what the agent could not source.

## When to run

- At the end of every epic, after the milestone write-up and before `/blog`.
- After any closed issue that changes how the data, splits, scorer, or runs
  work.

## Rules

- The guide holds no number without a link to its source.
- A gap the agent reports is fixed at the source, not guessed in the guide.
