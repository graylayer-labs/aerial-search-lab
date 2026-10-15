# Changelog

Key changes to the project, newest first. Routine fixes and wording changes
are left out; the git log has those. Entries are grouped by epic and date, and
each links to the issue or pull request that holds the detail.

## Epic: Domain adaptation (in progress)

### 2026-10-15

**New capability**
- `aerial-search sagemaker-train` runs one training run as a SageMaker
  managed-spot job on a T4 GPU: code pinned to a `main` commit and checked
  in the job and on fetch, data verified in the job, checkpoints in S3 so an
  interrupted job resumes, and a cost guard that refuses a run whose worst
  case passes EUR 15 or would take the month past EUR 60. No cloud run yet:
  AWS declined the on-demand GPU quota and the spot quota is pending.
  ([#113](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/113),
  [#128](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/128))

**Rule change**
- Cloud GPU runs no longer need a fresh owner approval each time: agents may
  spend up to EUR 15 per run and EUR 60 per month, on SageMaker managed spot
  only (AWS declined the on-demand quota).
  ([#125](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/125))

### 2026-10-14

**New capability**
- The fine-tuned detector saves checkpoints and its test predictions as it
  goes, and a stopped run resumes with the same result as an uninterrupted
  one, ready for spot instances; `aerial-search rescore` scores a finished
  run's predictions again, by person size, without the model.
  ([#112](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/112))
- The fine-tuned detector can now run on overlapping tiles at native
  resolution (`--input tiled`), so 4–16 px people are no longer shrunk away;
  boxes are stitched back to source pixels and merged before the unchanged
  scorer. No tiled result yet: the full grid takes about 26 h on the laptop
  and waits on a compute decision.
  ([#96](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/96),
  [#109](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/109))

## Epic: Off-the-shelf bar (in progress)

### 2026-10-04

**Result**
- At the same input resolution, an ordinary COCO-pretrained detector
  (Faster R-CNN, MobileNetV3) fine-tuned on the same frames beats the head
  on frozen SigLIP 2 features wherever labels are scarce: mean `ap_iou25` at
  10% is 0.163 against 0.113 (RGB) and 0.256 against 0.129 (thermal), and it
  is ahead in all six camera and fraction cells from 1% to 10%. At 100% the
  two are level, and the detector's boxes are tighter throughout. The
  fine-tuned detector, not the frozen head, is now the bar.
  ([results](docs/supervised-baseline-results.md),
  [#67](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/67))

**Found**
- The frozen-feature head peaks within 25 to 150 steps at 1% to 10% of labels
  and then overfits, but selecting the step every 25 steps instead of every
  250 moves mean `ap_iou25` by at most 0.01, within the spread. The first bar
  stands.
  ([#91](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/91))
- Resolution dominates: the same detector at full 4K resolution scores
  `ap_iou25` 0.779 on the RGB MtErie fold at 100%, against 0.213 at the
  cache's input size. One run.
  ([#67](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/67))

### 2026-10-03

**Result**
- Thermal does not need its own input handling yet. On the thermal view at
  10% and 100% of labels, a learned input stem changed mean `ap_iou25` by
  -0.009 and +0.007 against a matched control, within the spread over
  folds; histogram equalisation lowered it by 0.050 and 0.137, on three of
  four folds at each fraction. Copying the grey channel stays the default.
  ([results](docs/thermal-input-ablation.md),
  [#68](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/68))

**Found**
- 120 more training steps raised the 10%-label thermal heads from a mean
  `ap_iou25` of 0.129 to 0.162, while the 100% heads barely moved, so the
  first bar may stop small-label runs too early.
  ([#68](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/68))

**Changed**
- The owner waived the check of the 21 unsure label-completeness frames: the
  WiSARD boxes are used as given and the missing-label estimate stays a
  range, so measured precision is a floor.
  ([#37](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/37))

**New**
- The thermal input ablation can be run: `cache-features --thermal-input
  equalise` builds a separate equalised thermal cache, `train-head
  --input-handling stem` trains a learned input stem before the frozen
  backbone, and `head-table --ablation` tabulates the three arms per fold.
  Each run records its arm in `run.json`. `train-head --init-head <run>`
  warm-starts the stem arm and its matched control from a trained arm-A head,
  so the stem arm fits the laptop.
  ([#68](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/68))

### 2026-10-02

**Changed**
- The lead no longer reports or tallies usage; the owner sees it in their
  status line. The quiet limits stay: two agents, the cheaper model by
  default, one review per pull request, no new agents above 80%.
  ([#81](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/81))

**Result**
- First measured bar: a small head on frozen, unadapted SigLIP 2 features
  reaches a mean `ap_iou25` of 0.25 (RGB) and 0.30 (thermal) with all
  training labels, and 0.04 and 0.07 with 1%. `ap_iou50` stays under 0.09,
  people under 16 px are not found in RGB, and the spread over held-out
  sites is as large as the mean. One seed per run.
  ([results](docs/detection-head-results.md),
  [#66](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/66))

**Changed**
- The lead keeps the owner informed on a fixed rhythm: short updates when
  something finishes or after 30 minutes of silence, one ask at a time, a
  usage line, and reversible choices made alone and reported, not asked.
  ([#74](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/74))

**Added**
- `aerial-search train-head` trains a CenterNet-style person-centre head on
  the cached features, one camera, fold, label fraction and seed per run,
  and `head-table` gathers completed runs into a per-fold table with the
  mean and spread. Design and expected accuracy in
  `docs/detection-head-design.md`.
  ([#66](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/66))
- `aerial-search cache-features` runs the frozen SigLIP 2 NaFlex backbone once
  per image and caches pooled fp16 patch features, keyed by model, weight
  checksum, token budget, pooling and dtype, and resumable with a verified
  sample.
  ([#65](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/65))

**Found**
- One forward pass of SigLIP 2 at about 512 pixels takes 92 to 126 ms per
  image on the laptop, so a pooled feature cache for the whole dataset is
  about 11.5 GB and about an hour. 1024-pixel input is four to six hours and
  does not fit the disk. First cache: SigLIP 2 NaFlex, 1,024-token budget,
  pooled 2×2.
  ([#64](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/64),
  [note](docs/forward-pass-cost.md))

## Epic: Trusted ground (closing)

### 2026-10-01

**Changed**
- Thermal clip `210924_FHL_Enterprise_0403`: its 1,201 empty-label frames
  (about 86% of them show people with no box) are out of every thermal
  train, validation, and test set and stay in the unlabelled pool, through
  an override list in `folds.py` that `check-folds` verifies. The FHL thermal
  test set drops from 5,420 to 4,219 frames.
  ([#53](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/53))

**Added**
- The WiSARD dataset is pinned by a committed list of 100,794 files with
  size and SHA-256 (3.9 MB gzipped, root hash `6e5e5d55...75ed`). A normal
  `prepare` verifies the directories it reads against the list before
  writing, and a normal run re-verifies the directories its manifests
  reference before starting; `run.json` records what was verified. Manifests
  from `prepare --scratch` or from unpinned data cannot start a normal run.
  `aerial-search check-data` names every changed, missing or added file, and
  a directory fetched from S3 is verified before it is kept.
  ([#38](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/38))
- A follow-along guide to how the project was built, in `docs/guide/`, with five
  chapters on the first milestone, and a `guide-writer` agent and `/guide`
  skill that keep it current from closed issues. Every number in it links to
  its source. Short result-led blog posts are kept apart in `docs/blog/`, drafted
  with `/blog` for the owner to edit; none is written yet.
  ([#50](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/50),
  [guide](docs/guide/README.md))

**Changed**
- WiSARD is split by site-day, leaving one out: each of the four labelled
  site-days (MtErie, Carnation, FHL, Baker) is the test set of one fold, in
  three label views, and results are reported per fold and as mean and
  spread. The old random clip split, which put nothing in test, is gone. The
  unlabelled pool of a fold excludes the test site-day's images. Label
  fractions of 1, 5, 10, and 100% are nested blocks of 10 consecutive
  labelled frames. A checkpoint from one fold cannot start a normal run
  in another.
  `aerial-search check-folds` verifies that nothing of a test site-day
  reaches training.
  ([#3](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/3),
  [review](docs/site-folds-review.md))

**Found**
- First-pass estimate of unlabelled visible people (agent review, 99 frames
  per camera, 21 frames still unsure and awaiting the owner): about 1% of
  RGB frames (95% interval 0.2 to 6.1%) and 2% of thermal frames (0.6 to 7.9%)
  have an unboxed person, but the thermal misses sit in one clip,
  `210924_FHL_Enterprise_0403`, whose frames are mostly empty-labelled.
  ([#37](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/37),
  [review](docs/label-completeness-review.md))
- Frames 10 seconds apart in one clip are still far more alike than random
  frames of that clip; it takes about 50 seconds (250 frames). A split inside
  one clip now drops 250 frames between training and validation.
  ([#3](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/3))
- With four labelled site-days, 1% of a fold's training labels is 30 to 80
  frames (3 to 8 blocks of 10), sometimes from only two of the three
  training sites. A 1% result needs several seeds.
  ([#3](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/3))

**Decided**
- The starting backbone is SigLIP 2, kept frozen, with thermal fed to it
  unchanged as the first thermal experiment. The owner chose a non-Meta
  model; DINOv3 was the first recommendation. Licences for it and for candidate extra
  datasets were checked at their sources.
  ([#5](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/5),
  [review](docs/model-and-dataset-review.md))

### 2026-09-29

**Found**
- WiSARD frame pairing is fixed and checked. Frames are paired only within
  17 verified clip pairs from five flight days, giving 14,834 pairs, of which
  5,739 are labelled in both cameras. Both earlier manifest sets were wrong:
  one paired by position and included 1,620 mismatched pairs, the other lost
  most of seven clips to a parsing bug. One flight, Airfield, is excluded
  because its two cameras number frames at different rates. Pairs are 0 to 2
  frames apart in time, and this is accepted. A further 6,658 pairs are
  labelled in one camera only; new per-camera manifests carry them.
  ([#2](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/2),
  [review](docs/wisard-pairing-review.md))
- The code that pairs RGB frames with thermal frames reads only 5 or 6 digit
  frame numbers, and so misses about 53% of the image files in WiSARD. Every
  dataset figure quoted before this date is unverified. The bug is pinned by
  tests that are expected to fail until it is fixed.
  ([#2](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/2),
  [#12](https://github.com/graylayer-labs/ssl-aerial-person-detection/pull/12))
- The WiSARD paper reports that the two cameras drift by one or two frames
  over a flight, so equal frame numbers do not guarantee the same instant.
  ([#5](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/5))

**Added**
- Every experiment now starts through `aerial_search.run.start_run`, which
  refuses a run from a dirty tree or a commit not on `origin/main` and writes
  `run.json` (commit, config, seed, machine, input hashes, status). Results
  are written only into the run directory, and a normal run must start from a
  checkpoint made by a normal run. Debugging runs need `--scratch` and are
  labelled so their numbers cannot be quoted; only runs with status
  `completed` are quotable.
  ([#23](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/23))

**Changed**
- The dataset is stored once on the laptop and once in S3, with versioning on.
  81 GB of duplicate copies were deleted after a byte-for-byte comparison.
  Agents reach AWS through a role limited to the project bucket, which cannot
  delete.
  ([#7](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/7))
- The project restarted after five weeks idle, with a new direction: start
  from a pretrained foundation model, adapt it with self-supervised learning
  on unlabelled footage, then detect people with few labels. The earlier plan
  to train small networks from scratch is dropped.
  ([INTENT.md](INTENT.md), [docs/decisions.md](docs/decisions.md))
- `main` is protected. Changes arrive by pull request, merge themselves when
  CI passes, and are squashed. No human review is required.
  ([#10](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/10))
- Results may be quoted only from a run on a clean commit on `main`.
  ([#23](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/23))
- Tests and lint pass on `main`, and CI runs both on every pull request.
  ([#1](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/1))

**Added**
- A shared detection scorer. The primary metric is AP at IoU 0.25, because
  people are a few pixels across; recall at 0.01, 0.1 and 1 false alarms per
  image answers the search team's question. Results are split by modality,
  flight and person size, and do not depend on the order of the input when
  scores tie. Wherever `ap_iou25` is published, `ap_iou50` is shown beside it.
  ([#4](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/4),
  [review](docs/evaluation-metric-review.md))
- `INTENT.md` and `CLAUDE.md`: why the project exists, and how agents work in
  it.
  ([#10](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/10))
- A public project board, with six epics as the roadmap and tasks as their
  sub-issues.
  ([board](https://github.com/orgs/graylayer-labs/projects/1))
- Agents for implementing, reviewing, researching, auditing the process, and
  auditing the repo, each on the cheapest model that suits the work.
  ([#10](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/10),
  [#22](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/22))
- A cold-start test: a fresh agent with no context reports what the
  instructions failed to tell it. Two runs found and fixed fifteen gaps.
  ([#20](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/20))
- A process audit, with a standing log of its findings.
  ([#24](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/24))
- Archify, a third-party diagram skill for finished figures, installed by a
  script that checks a pinned commit and kept out of the repository. Its update
  check, brand fetch, and preview are blocked by settings. The README now shows
  the first figure, the pipeline.
  ([#30](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/30))

**Removed**
- The planning documents from the first attempt: `VISION.md`, `ROADMAP.md`,
  `ARCHITECTURE.md`, and `PROBLEM.md`. They remain in git history.
  ([#6](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/6))

**Decided against**
- Installing the Superpowers or Matt Pocock skill collections.
  ([review](docs/third-party-skills-review.md))
- Requiring signed commits on `main` while branch commits are unsigned. It
  blocked every pull request. Commits on `main` are still verified, because
  GitHub signs each squash merge.
  ([#22](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/22),
  [#27](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/27))

## Before the restart

### 2026-08-08 to 2026-08-26

- Project started. The WiSARD dataset was downloaded and paired, and a data
  exploration notebook was written. Its figures are unverified; see the first
  entry above.
- First experiment scripts for contrastive pretraining and grid detection.
  No results were recorded.
