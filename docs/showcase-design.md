# Showcasing the work

How this project gets seen outside the repo: when to post, how milestones
are shaped so each one has something to show, and how that is enforced.
Written for this repo first; the general parts are meant to move into the
shared setup later, so every project gets them.

## The problem

The work is documented well but nothing leaves the repo. Two blog posts sit
in [blog/](blog/) unpublished, graylayer.io has no page for the project, and
the README's "Where it's at" is a month stale. Someone has to find the repo
and read it to see any of this.

## Where it goes

| Channel | Role |
|---|---|
| graylayer.io | Home: the post and the project page live here |
| README | The proof behind the post; "Last published" date at the top |
| LinkedIn post | Distribution: ~150 words + picture + link, one per chapter |
| LinkedIn Featured | Pins the latest site post near the top of the profile |
| LinkedIn Projects | Filled in once; low visibility, not relied on |

## When to post

| Moment | Post | Where |
|---|---|---|
| Milestone closes (a result, good or bad) | Full post: claim first, one picture | Site, then LinkedIn |
| Surprise (a bug, a negative result, an "oh, that's why") | Short note | LinkedIn or a short site note |
| Visual moment (first working detection, a demo GIF) | Picture + two lines | LinkedIn |
| Project start, refactors, tooling | Nothing; fold into the next milestone post | — |

Post only when there is a new number or a new picture. Expect one every
2–4 weeks for an active project; a month with nothing to post means the work
has drifted, so look at the board, not the blog.

Collect as you go: save the figure into `docs/figures/` when you first see
it, and note the surprise in `docs/progress.md` the day it happens. At
milestone close the post is then mostly written.

## Designing work for it

| Level | Design question | Goes in |
|---|---|---|
| Project | What 3–4 headlines will this project let me post? | `INTENT.md`, "Posts this project will produce" |
| Milestone | What is the one claim and one picture at the end? | The milestone issue: `Claim:` and `Picture:` lines |
| Task | What does it leave behind for a post? | Done-when: a figure saved, or a line in `progress.md` |

- Show something early: a rough end-to-end result with a picture in the
  first week or two, then improve the slowest part.
- At most about 1 in 3 tasks is work nobody can see (refactor, tooling,
  setup) unless it blocks the next result.
- Milestones are 2–4 weeks. Longer means nothing to post.
- Parking a project gets a "where I left it" post. Written up counts as
  finished.

## This project's chapters

| Milestone | Claim | Picture |
|---|---|---|
| M2 (done) | A fine-tuned detector beats a frozen foundation model when labels are scarce | Bar chart (have); detections drawn on an RGB and a thermal frame (missing) |
| M3 | Cutting frames into tiles finds smaller people | Same frame before and after tiling, missed people circled |
| M4 | Self-supervised adaptation on unlabelled footage closes the gap, or does not | AP against share of labels, one line per method |

First chapter (about one session):

1. Detections from the fine-tuned detector on an RGB and a thermal frame, side by side.
2. Publish the two drafts in [blog/](blog/) on graylayer.io, plus a project page.
3. Update the README "Where it's at" and add "Last published".
4. LinkedIn: one post, pinned in Featured; a Projects entry.

## Enforcement

Block what can be blocked, automate the rest, and leave an approve step.

| Strength | Mechanism | How it works | Catches |
|---|---|---|---|
| Block | Milestone issue form | `Claim:` and `Picture:` are required fields | Milestones with nothing to show |
| Block | Action on milestone close | No `Published: <link>` in the closing comment: bot reopens and says why | Closing without posting |
| Block | CI check | A PR closing a milestone must add a `docs/blog/` file and update README "Last published" | Results never written up |
| Block | Claude Code hook | No headlines in `INTENT.md`: fix first. No post draft: a milestone task cannot close | Claude skipping it |
| Automate | Weekly scheduled agent | Finds unposted milestones, days since last post and the share of unseen work; drafts the post from `progress.md` and `docs/figures/` as a graylayer.io PR, plus LinkedIn text | Drift, and the effort of writing |
| Automate | Same agent: notification | "26 days quiet, 1 draft waiting" | Forgetting |
| Visible | graylayer.io projects page | Each card shows "last update" from the README | A project going quiet |

```
milestone issue --(form: Claim + Picture)--> work
      |
milestone PR --(CI: blog file + README date)--> merge
      |
close issue --(Action: Published link, else reopen)
      |
weekly agent --> draft site PR + LinkedIn text --> approve --> live
      `--> nudge if overdue
```

The weekly agent matters most: it does the writing, so publishing becomes an
approval rather than a chore.

Limits:

| Gap | Why | Mitigation |
|---|---|---|
| LinkedIn posting | No practical API for a personal profile | Agent drafts the text; paste it (~2 min) |
| Quality | A check sees that a post exists, not that it is good | Review the first three posts by hand |
| Opting out | Every check can be turned off | Public staleness on the site |

## Build order

1. First chapter above (content, no tooling).
2. Weekly drafting agent.
3. Milestone issue form and close Action.
4. CI check and Claude Code hook.
5. Move the general parts into the shared setup for every project.
