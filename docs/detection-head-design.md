# Detection head on cached features

Issue #66. The off-the-shelf bar: how far a frozen SigLIP 2 backbone gets
with 1, 5, 10 and 100% of the labels, per camera, on the four folds. Code:
`projects/frozen-head/src/frozen_head/head.py`,
`projects/frozen-head/src/frozen_head/head_experiment.py`,
`shared/aerial_search/src/aerial_search/evaluation/head_table.py`.

## What the cell size allows

The cache (docs/forward-pass-cost.md) holds one 768-d vector per 2×2 block
of 16-px patches of the resized image. In source pixels:

| camera, frame | model input | one patch | one cell |
|---|---|---|---|
| RGB 1920×1080 (most frames) | 672×384, 2.9× down | 46 px | 91 × 90 px |
| RGB 3840×2160 (all of MtErie) | 672×384, 5.7× down | 91 px | 183 × 180 px |
| thermal 640×512 | 560×448 | 18 px | 37 × 37 px (edge column 18 px) |

A 12-px person in a 1920-wide RGB frame is 4 input pixels: a quarter of one
patch. The head sees no pixels, only the cell vector, so all localisation
inside a cell must already be encoded in it.

How precise a box must be. For two equal squares of side `s` whose centres
differ by `d` along one axis, IoU = (s − d)/(s + d): IoU ≥ 0.25 needs
d ≤ 0.6 s, and IoU ≥ 0.5 needs d ≤ s/3. Shifted equally on both axes, the
limits are d ≤ 0.37 s and d ≤ 0.18 s. A perfectly centred box whose width and
height are both off by a factor r has IoU = 1/r², so 0.25 tolerates r ≤ 2
and 0.5 only r ≤ 1.41. For a 12-px RGB person, the centre must be right to
about 4 to 7 px at IoU 0.25 and 2 to 4 px at IoU 0.5: about 5% of a 91-px
cell, or one twentieth of a patch the model saw at 4 pixels wide. For a
20-px thermal person the same limits are 7 to 12 px and 4 to 7 px, a fifth
to a third of a cell.

**Expected before running anything.** RGB will be poor: `ap_iou25` around
0.05 to 0.2 at 100% labels, `ap_iou50` well below it, and very_tiny people
(under 8 px) near zero; the score will come from the small and medium
people. Thermal will be clearly better, `ap_iou25` perhaps 0.2 to 0.5, with
`ap_iou50` closer to it because a thermal cell is five times smaller. The gap
between the 1% and 100% rows will be large for both. A weak RGB number is a
finding about frozen features at this resolution, not a bug to chase.

## Design and precedent

A CenterNet head (Zhou, Wang and Krähenbühl, *Objects as Points*, 2019,
arXiv:1904.07850) on a grid four times finer than the feature grid. CenterNet
predicts, per output cell, a centre heatmap, a sub-cell offset and a box size,
and decodes peaks without anchors or NMS. It is the simplest detector with a
direct answer to "where in the cell is the person", which is the whole
problem here; anchor-free FCOS (arXiv:1904.01355) would need the same
sub-cell regression plus centre-ness and more assignment rules. Detecting
from a single-scale plain-ViT map with a light upsampling head follows ViTDet
(Li et al., arXiv:2203.16527). The finer grid comes from a sub-pixel
convolution (pixel shuffle; Shi et al., ESPCN, arXiv:1609.05158), which adds
no information but lets nearby people get separate peaks.

`CentreHead`: LayerNorm over the 768 channels, 1×1 conv to 256, GELU, 3×3
conv, GELU, 1×1 conv to 5·u² channels, pixel shuffle by u = 4. Out come a
centre logit, an offset (sigmoid, 0 to 1) and a log size (width and height)
per sub-cell. About 0.8 M parameters. The centre bias starts at a prior of
0.01, as in RetinaNet, so the sparse positives are not swamped early.

## Targets

- Boxes come from the fold's per-camera manifest (normalised centre and size)
  times the manifest's image size, which must equal the cache's.
- Sub-cell edges come from `cell_boxes` (source-pixel box of each cell), each
  cell split into four equal parts each way. The narrow edge column of
  thermal (one patch, 18 px) is split the same way, so its sub-cells are
  4.6 px wide.
- A person sets 1 at the sub-cell holding its centre (a centre exactly on
  the far image edge stays in the last sub-cell), its offset inside that
  sub-cell, and log(width / full cell width), log(height / full cell height).
  Sizes use the full cell, not the local one, so a narrow edge cell does not
  change a person's size target, and a 4K frame and a 2K frame of the same
  scene get the same targets.
- No Gaussian splat. CenterNet spreads each peak by a radius set by the box
  size; here every person is smaller than a sub-cell, so the radius is zero
  and the target is one cell.
- Two people in one sub-cell: the first keeps it, the other is counted in
  `training.json` under `targets_lost_to_shared_subcells`.
- A frame in a labelled manifest with an empty `boxes` list is a negative.
  A record without `boxes` is refused, never read as empty. Frames not in
  the manifest (the unlabelled pool, or the 99% hidden at 1%) are never read.

## Loss

CenterNet's: focal loss on the centre map (α = 2; with one-cell peaks its β
term is 1 everywhere), plus L1 on the offset and L1 on the log size at each
person's sub-cell, all divided by the number of people in the batch, weights
1, 1, 1. CenterNet weighs size by 0.1 because it regresses raw pixels; log
size is already on the scale of the offset.

## Training recipe

- AdamW, learning rate 1e-3, weight decay 1e-4, cosine decay to zero.
- 2,000 steps of batch 32 at every fraction, seed 7. At 1% (30 to 70 frames)
  that is about a thousand passes over the data; validation picks the step.
- Validation `ap_iou25` every 250 steps and at the last step; the best step's
  weights are scored once on test. A NaN score (no people) never wins.
- The step is chosen on the fold's **full** labelled validation split at
  every fraction. So "1% of labels" means 1% of the training labels plus
  all the validation labels: RGB with test site-day MtErie trains on 70
  frames at 1% and selects on 1,339. `head-table` prints this under the table.
- No augmentation: the features are cached, and a flipped image does not
  give flipped features.
- Decoding: sigmoid, a sub-cell is a peak if it is the maximum of its 3×3
  neighbourhood, the 100 highest peaks per image become boxes, clipped to the
  image. Scored by `aerial_search.evaluation.detection`; `ap_iou50` is
  reported beside `ap_iou25`.

## Runs and safeguards

```bash
uv run aerial-search train-head rgb --fold 210417_MtErie --percent 10
uv run aerial-search head-table outputs/detection-head/siglip2-base-naflex-1024tok
```

- One run is one camera, fold, fraction and seed; it writes `run.json`,
  `detection_metrics.json` (test), `training.json` and `head.pt` under
  `outputs/detection-head/<cache>/<camera>-<fold>-<p>pct-seed<seed>/`.
- `run.json` hashes the three manifests, the cache's `cache.json` and its
  `index.jsonl`, and lists the cache sessions the features came from.
- Train and validation records may not come from the test site-day (by the
  record's `site_day` and by its image path); test records must.
- A normal run refuses a cache under `scratch-features/` or holding a
  `session.lock`, features written by a cache session that is scratch, not
  completed, or without a run record, a source hash that is not the pinned
  one, and an incomplete cache. A scratch run leaves missing frames out and
  lists them in `run.json`.
- `head-table` refuses scratch, unfinished, or duplicate runs, and runs whose
  recipe, seed or `cache.json` hash differ. It records each fold's commit and
  warns if commits differ, and gives beside each size-bucket mean the number
  of folds that had people of that size.

## Limits of this first pass

One person per sub-cell, and one peak per 3×3 neighbourhood. People who share
a sub-cell are lost at target assignment. People in touching sub-cells
compete in the 3×3 peak test, which keeps one of two neighbours. Counts on
the test manifests, by `uv run python tools/head_crowding.py
outputs/features/siglip2-base-naflex-1024tok data/manifests/wisard-full`.
"Adjacent" counts people whose sub-cell touches another occupied one;
"unreachable" counts the fewest of those the peak test must lose:

| camera | test site-day | people | share a sub-cell | adjacent | unreachable |
|---|---|---|---|---|---|
| RGB | MtErie | 1,770 | 1.3% | 19.5% | 9.3% |
| RGB | Carnation | 7,189 | 1.2% | 22.5% | 10.5% |
| RGB | FHL | 7,466 | 1.8% | 11.2% | 5.6% |
| RGB | Baker | 6,012 | 1.4% | 2.3% | 1.1% |
| RGB | all | 22,437 | 330 (1.5%) | 2,932 (13.1%) | 1,405 (6.3%) |
| thermal | MtErie | 1,824 | 0.8% | 14.1% | 6.5% |
| thermal | Carnation | 1,714 | 14.0% | 60.6% | 26.1% |
| thermal | FHL | 8,079 | 1.8% | 7.5% | 3.7% |
| thermal | Baker | 5,261 | 0.2% | 2.2% | 1.1% |
| thermal | all | 16,878 | 405 (2.4%) | 2,020 (12.0%) | 925 (5.5%) |

So recall on thermal Carnation cannot pass about 60%, whatever the
features. Each run writes these counts for its splits to `training.json`
(`crowding`), with the test's tied-score predictions.

Also: the same step count at every fraction; one grid shape per camera
(true of the cache today); no learning-rate search.
