---
name: handoff
description: End a session by writing the current state onto the GitHub issue so the next agent can continue. Use before stopping work, finished or not.
---

# Handoff

Write down what the next agent needs. Assume they have never seen this
conversation.

## Steps

1. Make sure the work is on a branch and committed in logical units. Do not
   commit broken code to make the tree clean; say in the handoff what is
   uncommitted and why.
2. Run `uv run pytest` and `uv run ruff check .` and note the result.
3. Draft a comment for the issue using the template below.
4. Post it with `gh issue comment <number>`, before the issue closes. If a
   merge closed the issue first, post it straight after.
5. Update the issue's status on the board if it changed.
6. If the session surfaced follow-up work, create issues for it from the task
   template and add them to the board.
7. Tell the owner with an update in the form of the `owner-update` skill: what finished and the one
   thing that needs them, if any. Open anything they need to look at.

## Comment template

```markdown
## Handoff <date>

**State:** done | in progress | blocked

**Branch:** `<branch>` (pushed: yes/no)

**Done so far**
- ...

**Evidence**
- Commands run and their output, pasted as it appeared. For a result, the
  numbers with the config and seed that produced them.

**Decisions and why**
- ...

**Tried and did not work**
- What, and the most likely reason.

**Next step**
- The single next action, specific enough to start on.

**Open questions for the owner**
- ...

```

## When the issue is finished

Use the same template with **State: done**, and add:

- a link to the merged PR
- what was found but left alone, and the issue that now tracks it

Then add one line to the parent epic's "Decisions so far" list: the date, the
outcome in a sentence, and a link to this issue.

Then leave a note for the guide. Append a file `docs/guide/notes/<issue>.md`
of ten lines or fewer: what the issue set out to do, what it found, the
decision and why, and which sources hold the numbers. No number goes in the
note unless the source is named beside it. This is cheap, and it means the
guide and the blog can be assembled later without re-reading every issue. Do
it in the same PR as the change when there is one; otherwise commit it alone.

## Rules

- Write for an agent that has never seen this project.
- Write whole sentences. Do not use shorthand or labels coined during the
  session.
- Include failures. A dead end that is not written down gets repeated.
