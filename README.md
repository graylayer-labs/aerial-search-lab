# Aerial person detection for search and rescue

[![ci](https://github.com/graylayer-labs/ssl-aerial-person-detection/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/graylayer-labs/ssl-aerial-person-detection/actions/workflows/ci.yml)

A personal research lab, run like a thesis: one question, several
approaches, one honest evaluation. The question is how to find people in
drone footage from RGB (ordinary) and thermal (infrared) cameras when footage
is plentiful and labels, the boxes drawn by hand, are scarce. Each approach
lives in its own folder under [projects/](projects/), and all of them are
scored by the same shared evaluation in [shared/](shared/aerial_search/): the
same data, the same held-out site-days, the same metrics and the same record
of how every run was made. Failures are written up with the same care as
successes.

## Why

Search and rescue teams record far more footage than anyone can label, so a
detector that needs a large labelled dataset is of little use to them. The two
cameras also fail differently: thermal picks up warm rocks and animals, RGB
loses people in darkness and under foliage. The plan is to start from a
pretrained vision model, adapt it with self-supervised learning (learning from
unlabelled footage), train a person detector on a small share of the labels,
and measure everything on flights the model has never seen.
[INTENT.md](INTENT.md) has the purpose and the standard.

![Pipeline diagram. RGB and thermal drone footage, mostly unlabelled, is paired frame by frame. A pretrained vision foundation model is adapted to the paired footage with self-supervised learning. A person detector is trained on top using a small labelled fraction, then evaluated on flights the model has never seen.](docs/figures/pipeline.svg)

## Approaches

Mean over the four held-out site-days at 10% of the training labels, as
`ap_iou25` / `ap_iou50` (explained under Results so far).

| Project | Approach | Status | RGB | Thermal | Source |
|---|---|---|---|---|---|
| [finetuned-detector](projects/finetuned-detector/) | A COCO-pretrained Faster R-CNN fine-tuned on the few labels: the bar | Results in; tiled runs next | 0.163 / 0.046 | 0.256 / 0.137 | [results](docs/supervised-baseline-results.md) |
| [frozen-head](projects/frozen-head/) | A small head on frozen SigLIP 2 features | Results in; below the bar | 0.113 / 0.006 | 0.129 / 0.007 | [results](docs/supervised-baseline-results.md) |

## Where it's at (2026-10-23)

- **Done:** the data is paired, split by site and day, and scored; the
  baseline to beat is an ordinary fine-tuned detector, which leads a frozen
  foundation model at 1% to 10% of labels ([Milestone 2](docs/milestones/off-the-shelf-bar.md)).
- **Done:** training runs can be sent to a cloud GPU as cheap interruptible
  jobs, capped at EUR 15 a run and EUR 60 a month. No job has run yet.
- **Done:** the detector can read frames at full resolution by cutting them into
  tiles, and a stopped run resumes. No tiled score yet.
- **Done:** the repo is reshaped into a lab of approaches that share one
  evaluation ([#160](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/160)).
- **Doing:** the first SageMaker spot job ([#127](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/127)).
- **Next:** cloud runs of the tiled detector ([#96](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/96)), then adapting the model with unlabelled footage.

The dated log is [docs/progress.md](docs/progress.md).

## Results so far

The best result so far is the fine-tuned detector. Scores are mean AP over four
held-out site-days, shown as `ap_iou25` / `ap_iou50` (a box counts as right at
a quarter or half overlap with the true box), one seed per run. Per-fold
tables and spread are in the linked results.

| Claim | Number | Source |
|---|---|---|
| A fine-tuned detector beats a frozen foundation-model head with few labels (RGB, 10% of labels) | 0.163 / 0.046 against 0.113 / 0.006 | [Milestone 2](docs/milestones/off-the-shelf-bar.md), [results](docs/supervised-baseline-results.md) |
| Same, thermal, 10% of labels | 0.256 / 0.137 against 0.129 / 0.007 | [Milestone 2](docs/milestones/off-the-shelf-bar.md), [results](docs/supervised-baseline-results.md) |
| With all labels neither is clearly ahead on `ap_iou25`, but the detector's boxes are tighter (RGB; thermal) | 0.204 / 0.060 against 0.250 / 0.056; 0.294 / 0.176 against 0.302 / 0.083 | [Milestone 2](docs/milestones/off-the-shelf-bar.md) |
| Resolution is the largest lever found (RGB, MtErie fold, 100%, one run) | 0.779 / 0.448 at full frame against 0.213 / 0.065 shrunk | [results](docs/supervised-baseline-results.md) |
| Thermal needs no special input handling: equalising contrast hurts | -0.050 (10%) and -0.137 (100%) in mean `ap_iou25` | [ablation](docs/thermal-input-ablation.md) |
| The pairing is verified | 14,834 RGB and thermal pairs from 17 clip pairs | [Milestone 1](docs/milestones/trusted-ground.md), [review](docs/wisard-pairing-review.md) |

![Mean test AP over four held-out site-days, frozen SigLIP 2 plus a head against a fine-tuned detector, RGB and thermal](docs/figures/off-the-shelf-bar.svg)

## How to read this repo

Start with `docs/progress.md`, then the two milestones.

**Code**
- `shared/aerial_search/`: the shared evaluation (data, folds, frame
  selection, metrics, tables, run provenance, cloud runs) and the
  `aerial-search` command.
- `projects/<name>/`: one approach each, with its own README.
- `tools/`: one-off scripts worth keeping.

**Story**
- `docs/progress.md`: what is now true, newest first.
- `docs/milestones/`: one write-up per finished epic (a group of related tasks).
- `docs/guide/`: a step-by-step account of how it was built, mistakes included.
- `docs/blog/`: short result-led posts.
- `CHANGELOG.md`: key changes in two minutes.
- `docs/decisions.md`: every technical choice, one line each.

**Results**
- `docs/detection-head-results.md`: the frozen-feature head, per fold.
- `docs/supervised-baseline-results.md`: the fine-tuned detector, per fold.
- `docs/thermal-input-ablation.md`: three ways to feed thermal frames in.
- `docs/figures/`: the pipeline diagram and the Milestone 2 chart.

**Design**
- `docs/detection-head-design.md`: how the head is built.
- `docs/forward-pass-cost.md`: what one model pass costs on the laptop.

**Reviews** (the evidence behind a decision)
- `docs/wisard-pairing-review.md`: how RGB and thermal frames are paired.
- `docs/site-folds-review.md`: how the data is split.
- `docs/evaluation-metric-review.md`: why the metric is `ap_iou25`.
- `docs/label-completeness-review.md`: how many people have no box.
- `docs/model-and-dataset-review.md`: why SigLIP 2.
- `docs/third-party-skills-review.md`: outside agent skills considered.

**Rules:** [INTENT.md](INTENT.md) is why; [CLAUDE.md](CLAUDE.md) is how.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev      # the workspace: shared/ and every project
uv run pytest
uv run ruff check .
uv run aerial-search --help
```

## Starting a new approach

Copy `projects/_template`, write the expected results before the first run,
and register its commands. The steps are in [projects/README.md](projects/README.md).

## How this project is run

Claude agents do the engineering and the owner directs. Work is planned as
[epics and issues](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues?q=is%3Aissue+label%3Aepic),
each written so that an agent with no prior context can pick it up.

## Licence

MIT. See [LICENSE](LICENSE).
