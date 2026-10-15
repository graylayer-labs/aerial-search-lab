# Progress

What is now true, newest first, one entry per finished piece of work. Each
entry says what changed, why it matters and where the detail is. For the
current state and the headline results, start with the [README](../README.md).
Short terms: **AP** (average precision) scores how well a detector finds
people, from 0 to 1, higher is better. **`ap_iou25`** counts a box as right if
it overlaps the true box by at least a quarter; **`ap_iou50`** needs half.
A **fold** is one test: it holds out one site on one day that the model never
trained on. **Frozen** means the model's weights are not changed.

## 2026-10-15 · A training run can now be sent to a cloud GPU, with a spending cap

Before, every run needed the laptop open. `aerial-search sagemaker-train` now
sends one run to an AWS GPU as a cheap interruptible ("spot") job, and
`sagemaker-fetch` brings the result back and re-checks it. A run is refused
unless its worst case stays under EUR 15, and the month under EUR 60. Code is
pinned to a commit on `main`, the data is verified inside the job, and
checkpoints go to S3 so an interrupted job resumes. Eleven small pieces, merged
together: archive and hash the source, price a job, build the request, verify
and install inside the container, launch with a dry run, fetch, and the two
commands.
Result: none yet. AWS declined the on-demand GPU quota and the spot quota is
pending, so no job has run in the cloud.
Links: [#113](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/113), [#128](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/128), [CHANGELOG](../CHANGELOG.md),
[`supervised-baseline-results.md`](supervised-baseline-results.md), next step [#127](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/127).

## 2026-10-15 · Another model's saved predictions are set aside

A saved predictions file that belongs to a different model is now set aside
instead of being used. Detail in the issue.
Links: [#130](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/130).

## 2026-10-14 · The fine-tuned detector can read people at full resolution

Shrinking a 4K frame to the model's input size leaves a person 4 to 16 pixels
across a pixel or two. `train-baseline --input tiled` cuts each frame into
overlapping tiles at native resolution, runs the detector, and merges the boxes
back to source pixels before the unchanged scorer. Training and scoring also
checkpoint, so a stopped run resumes with the same result, and `rescore` scores
a finished run's predictions again by person size without the model.
Result: no tiled score yet. The full grid takes about 26 h on the laptop and
waits on cloud runs.
Links: [#96](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/96), [#109](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/109), [#110](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/110), [#112](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/112), [CHANGELOG](../CHANGELOG.md),
[`supervised-baseline-results.md`](supervised-baseline-results.md).

## 2026-10-14 · Pull requests are checked by the organisation's shared CI

CI (the automatic checks on every pull request) now uses the organisation's
shared workflow and PR gate instead of a copy kept in this repo. Nothing
visible in the results.
Links: [#107](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/107), [#120](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/120).

## 2026-10-05 · CI actions are pinned

The outside actions CI uses are pinned to fixed versions with least-privilege
permissions, so an upstream change cannot alter what runs. Nothing visible in
the results.
Links: [#103](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/103).

## 2026-10-04 · Milestone 2 closed: an ordinary fine-tuned detector is the bar

A frozen foundation model was measured with 1% to 100% of the labels, and an
ordinary detector fine-tuned on the same labels beat it where labels are
scarce. This sets the number every later technique has to beat.
Result: at 10% of labels, mean `ap_iou25` / `ap_iou50` is 0.163 / 0.046
against 0.113 / 0.006 (RGB) and 0.256 / 0.137 against 0.129 / 0.007 (thermal),
over four folds, one seed per run. At 100% the two are level.
Links: [write-up](milestones/off-the-shelf-bar.md), [#69](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/69),
![Mean test AP over four held-out site-days](figures/off-the-shelf-bar.svg)

## 2026-10-04 · The fine-tuned detector beats the frozen head with few labels

A COCO-pretrained Faster R-CNN (MobileNetV3), trained on the same frames at the
same input size, is ahead of the head on frozen SigLIP 2 features at 1%, 5% and
10% of labels on both cameras, and level at 100%. Its boxes are tighter
throughout. One run at full 4K resolution scored `ap_iou25` 0.779 /
`ap_iou50` 0.448 on the MtErie fold at 100%, against 0.213 / 0.065 at the
shared input size: resolution is the largest lever found.
Links: [#67](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/67), [`supervised-baseline-results.md`](supervised-baseline-results.md).

## 2026-10-04 · The frozen head was not stopped too early

The head peaks within 25 to 150 steps at 1% to 10% of labels and then
overfits. Choosing the step more finely moves mean `ap_iou25` by at most
0.01, within the spread over folds, so the first bar stands.
Links: [#91](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/91), [`detection-head-results.md`](detection-head-results.md).

## 2026-10-03 · Thermal frames need no special handling

Copying the grey channel into the three colour channels is as good as a learned
input stem (-0.009 at 10% and +0.007 at 100% in mean `ap_iou25`, against a
matched control). Equalising contrast hurts (-0.050 and -0.137).
Links: [#68](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/68), [#86](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/86), [#88](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/88), [`thermal-input-ablation.md`](thermal-input-ablation.md).

## 2026-10-03 · The labels are used as given

The owner chose to use the WiSARD boxes as given and skip the check of the 21
frames the agents could not decide on. The estimate of unboxed people stays a
range, so measured precision is a floor.
Links: [#37](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/37), [`label-completeness-review.md`](label-completeness-review.md).

## 2026-10-02 · First measured bar: a small head on frozen SigLIP 2 features

A small head trained on frozen, unadapted SigLIP 2 features reaches a mean
`ap_iou25` / `ap_iou50` of 0.250 / 0.056 (RGB) and 0.302 / 0.083 (thermal) with
all labels, and 0.044 / 0.001 and 0.065 / 0.001 with 1%. People under 16 pixels
are not found in RGB. The spread over folds is as large as the mean.
Links: [#66](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/66), [`detection-head-results.md`](detection-head-results.md),
[`detection-head-design.md`](detection-head-design.md).

## 2026-10-02 · Features are computed once and reused

`cache-features` runs the frozen model once per image and stores the result, so
many experiments share one pass. One forward pass takes 92 to 126 ms per image
on the laptop; the cache for the whole dataset is about 11.5 GB and about an
hour. Frames with no partner in the other camera are cached too.
Links: [#64](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/64), [#65](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/65), [#78](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/78), [`forward-pass-cost.md`](forward-pass-cost.md).

## 2026-10-02 · The lead keeps the owner informed without being asked

The lead sends short updates when something finishes, and no longer reports
usage; the owner sees it in their status line.
Links: [#74](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/74), [#81](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/81).

## 2026-10-01 · Milestone 1 closed: Trusted ground

Before training anything, the pairing, the test split, the scoring and the
reproducibility of every result were checked. Most needed fixing.
Result: 14,834 verified RGB and thermal pairs from 17 clip pairs, four folds
that each leave one site-day out, and a run that refuses to start from
unverified code or data.
Links: [write-up](milestones/trusted-ground.md), [#8](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/8), [#71](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/71),
[blog draft](blog/2026-10-trusted-ground.md).

## 2026-10-01 · Results are tested on flights the model has not seen

WiSARD is split by site-day, leaving one out: MtErie, Carnation, FHL and Baker
each test one fold, and results are reported per fold with the mean and
spread. Label fractions of 1%, 5%, 10% and 100% are nested. A check confirms
nothing of a test site-day reaches training. With so few site-days, 1% of
labels is 30 to 80 frames, so a 1% result needs several seeds.
Links: [#3](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/3), [#45](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/45), [`site-folds-review.md`](site-folds-review.md).

## 2026-10-01 · A run says which data it used

The dataset is pinned by a committed list of 100,794 files with size and
SHA-256. A normal run re-checks the directories it reads against the list, and
`run.json` records what was verified.
Links: [#38](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/38), [`CLAUDE.md`](../CLAUDE.md#pinning-the-data).

## 2026-10-01 · One thermal clip is mostly unlabelled and is kept out of testing

In clip `210924_FHL_Enterprise_0403`, about 86% of the 1,201 empty-label frames
show people with no box. They are out of every thermal train, validation and
test set; the FHL thermal test set drops from 5,420 to 4,219 frames. A first
estimate puts unboxed people at about 1% of RGB frames and 2% of thermal frames.
Links: [#49](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/49), [#53](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/53), [`label-completeness-review.md`](label-completeness-review.md).

## 2026-10-01 · The starting model is SigLIP 2

SigLIP 2 (Google, Apache-2.0), kept frozen to begin with. The owner chose a
non-Meta model; EVA-02 is second and DINOv2 the comparison.
Links: [#5](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/5), [`model-and-dataset-review.md`](model-and-dataset-review.md).

## 2026-10-01 · Every commit is signed; a follow-along guide exists

All commits, on branches as well as `main`, are signed with a project key. A
guide in `docs/guide/` explains how the project was built, with every number
linked to its source.
Links: [#27](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/27), [#50](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/50), [guide](guide/README.md).

## 2026-09-29 · The frame pairing is fixed and checked

The earlier pairing was wrong in two ways: one set paired by position and
included 1,620 mismatched pairs, the other lost most of seven clips to a parser
that read only 5 or 6 digit frame numbers. Frames are now paired only within 17
verified clip pairs from five flight days: 14,834 pairs, 5,739 labelled in
both cameras. Every dataset figure from before this date is unverified.
Links: [#2](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/2), [`wisard-pairing-review.md`](wisard-pairing-review.md).

## 2026-09-29 · People can be scored fairly

A shared detection scorer uses AP at an overlap of 0.25 as the main metric,
because people are a few pixels across, with `ap_iou50` beside it, and recall at
fixed false alarms per image. Results are split by camera, flight and person
size.
Links: [#4](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/4), [`evaluation-metric-review.md`](evaluation-metric-review.md).

## 2026-09-29 · A quoted result can be traced to its code

Every experiment starts through one function that refuses a dirty tree or a
commit not on `main`, and writes `run.json` with the commit, config, seed and
machine. Only runs with status `completed` are quotable.
Links: [#23](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/23), [`CLAUDE.md`](../CLAUDE.md#experiments).

## 2026-09-29 · The project restarted with a clear direction and working process

After five weeks idle, the project restarted: a pretrained foundation model,
adapted with self-supervised learning, then few-label detection. Tests and lint
pass on a protected `main`. The dataset is stored once on the laptop and once in
S3, freeing 81 GB. Agents, a public board, a changelog, process audits and a
cold-start test were set up.
Links: [#1](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/1), [#6](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/6), [#7](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/7), [#10](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/10), [#20](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/20), [#22](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/22), [#24](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/24), [#28](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/28), [#29](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/29), [#30](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/30),
[`INTENT.md`](../INTENT.md), [`decisions.md`](decisions.md).
