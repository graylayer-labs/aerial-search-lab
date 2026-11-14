---
name: blog
description: Draft a blog post for a milestone, from the guide and the milestone note, for the owner to edit. Use after /guide at the end of an epic. Never publishes.
---

# Blog

## Steps

1. Take the milestone name from the owner or the epic. Check that
   `docs/milestones/<name>.md` exists. If it does not, stop: the post comes
   from the milestone note, so write the note first.
2. Check that `/guide` has been run since the milestone's last issue closed.
   If not, run it first.
3. Delegate the draft to the `guide-writer` agent, naming the milestone. It
   reads the note, the guide chapters, and `docs/blog/TEMPLATE.md`, and writes
   `docs/blog/<date>-<slug>.md`.
4. Read the draft. Check the shape: one-paragraph hook, one figure, the
   result, what it means for search and rescue, a link to the guide. Trace
   five numbers to their sources. Check that `ap_iou50` sits beside every
   `ap_iou25` and that results are per fold.
5. The post needs one finished figure. Make it with Archify, as `docs/knowledge/recipes.md`
   describes under "Make a finished diagram", or ask the owner what they want shown.
6. Open the draft for the owner (`open <file>`), and put its hook and its
   headline result in your message. Ask for their edits.
7. Commit on a branch, as the `flow-coding` playbook describes, and open a PR for the draft.

## Rules

- The post is a draft for the owner. Do not publish it or post it anywhere.
  Publishing is the owner's (`CLAUDE.md`, Repo rules).
- No number without a source in the guide or a note.
- Report negative results as plainly as positive ones.
