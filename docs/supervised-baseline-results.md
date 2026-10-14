# Supervised baseline against the frozen backbone: results

"A frozen foundation model with few labels" means little unless it beats the
ordinary approach: an off-the-shelf detector fine-tuned on the same labels.
This compares the two on every fold, camera and label fraction (issue #67).
The frozen-backbone results are in `detection-head-results.md`.

## Answer

At the same input resolution, the ordinary approach wins where labels are
scarce. A COCO-pretrained Faster R-CNN (MobileNetV3) fine-tuned on the same
frames beats the head on frozen SigLIP 2 features at 1%, 5% and 10% of labels,
on both cameras. At 100% the two are level on `ap_iou25`. The detector's boxes
are much tighter at every fraction (`ap_iou50`).

So the frozen-feature head is not the bar to beat. The supervised detector is.

Resolution matters more than either method. The one full-resolution run (RGB,
MtErie, 100%) scores `ap_iou25` 0.779 against 0.213 for the same detector at
the cache's input size.

## Results

Test `ap_iou25` / `ap_iou50` per fold, mean ± sd over the four folds. "head" is
the #66 head on frozen SigLIP 2 features; "baseline" is the fine-tuned
detector at the cache's input size (672×384 RGB, 560×448 thermal), the fair row.

### RGB

| Labels | Method | MtErie | Carnation | FHL | Baker | `ap_iou25` | `ap_iou50` |
|---|---|---|---|---|---|---|---|
| 1% | head | 0.031 / 0.001 | 0.079 / 0.003 | 0.026 / 0.002 | 0.038 / 0.001 | 0.044 ± 0.024 | 0.001 ± 0.001 |
| 1% | baseline | 0.016 / 0.000 | 0.101 / 0.011 | 0.049 / 0.018 | 0.092 / 0.007 | 0.064 ± 0.039 | 0.009 ± 0.007 |
| 5% | head | 0.062 / 0.001 | 0.089 / 0.004 | 0.082 / 0.009 | 0.144 / 0.004 | 0.094 ± 0.035 | 0.004 ± 0.003 |
| 5% | baseline | 0.152 / 0.039 | 0.157 / 0.049 | 0.089 / 0.035 | 0.209 / 0.079 | 0.152 ± 0.049 | 0.051 ± 0.020 |
| 10% | head | 0.077 / 0.002 | 0.115 / 0.008 | 0.045 / 0.002 | 0.216 / 0.012 | 0.113 ± 0.075 | 0.006 ± 0.005 |
| 10% | baseline | 0.167 / 0.036 | 0.190 / 0.052 | 0.102 / 0.036 | 0.192 / 0.058 | 0.163 ± 0.042 | 0.046 ± 0.011 |
| 100% | head | 0.226 / 0.030 | 0.150 / 0.026 | 0.086 / 0.007 | 0.540 / 0.163 | 0.250 ± 0.201 | 0.056 ± 0.072 |
| 100% | baseline | 0.213 / 0.065 | 0.201 / 0.061 | 0.099 / 0.038 | 0.302 / 0.077 | 0.204 ± 0.083 | 0.060 ± 0.016 |

### Thermal

| Labels | Method | MtErie | Carnation | FHL | Baker | `ap_iou25` | `ap_iou50` |
|---|---|---|---|---|---|---|---|
| 1% | head | 0.063 / 0.001 | 0.000 / 0.000 | 0.028 / 0.003 | 0.168 / 0.002 | 0.065 ± 0.073 | 0.001 ± 0.001 |
| 1% | baseline | 0.130 / 0.048 | 0.002 / 0.000 | 0.073 / 0.026 | 0.406 / 0.214 | 0.153 ± 0.177 | 0.072 ± 0.097 |
| 5% | head | 0.177 / 0.004 | 0.001 / 0.000 | 0.076 / 0.012 | 0.276 / 0.012 | 0.132 ± 0.120 | 0.007 ± 0.006 |
| 5% | baseline | 0.256 / 0.107 | 0.009 / 0.001 | 0.162 / 0.062 | 0.482 / 0.279 | 0.228 ± 0.198 | 0.112 ± 0.119 |
| 10% | head | 0.229 / 0.010 | 0.001 / 0.000 | 0.121 / 0.011 | 0.165 / 0.008 | 0.129 ± 0.096 | 0.007 ± 0.005 |
| 10% | baseline | 0.311 / 0.129 | 0.002 / 0.000 | 0.186 / 0.078 | 0.527 / 0.341 | 0.256 ± 0.220 | 0.137 ± 0.146 |
| 100% | head | 0.313 / 0.057 | 0.003 / 0.000 | 0.202 / 0.030 | 0.692 / 0.247 | 0.302 ± 0.290 | 0.083 ± 0.111 |
| 100% | baseline | 0.368 / 0.180 | 0.007 / 0.000 | 0.220 / 0.114 | 0.580 / 0.410 | 0.294 ± 0.242 | 0.176 ± 0.173 |

### Full resolution (second row, one run)

| Camera, fold, labels | baseline at cache size | baseline at full resolution | head |
|---|---|---|---|
| RGB, MtErie, 100% | 0.213 / 0.065 | 0.779 / 0.448 | 0.226 / 0.030 |

MtErie is the only 4K site-day, so it is where resizing to 672×384 loses the
most. The other seven full-resolution runs were skipped: this one took 51
minutes, over the 30-minute limit set for the row.

## Reading the results

- **Few labels: the detector wins.** At 1% to 10% it is ahead on mean
  `ap_iou25` in all six camera–fraction cells, and per fold in 22 of 24. The
  two exceptions are RGB MtErie at 1% and RGB Baker at 10%; three of the 22
  are thermal Carnation, where both methods score near zero. In thermal at
  10% it doubles the head (0.256 against 0.129).
- **All labels: level.** At 100% the means are within the spread (RGB 0.204
  against 0.250; thermal 0.294 against 0.302). The head's lead comes from one
  fold, Baker.
- **Boxes: the detector wins everywhere.** The head's `ap_iou50` is under 0.09
  at every fraction; the detector's is up to 0.18. The head predicts a centre
  and a size on a coarse grid; the detector regresses boxes at its own scale.
- **This is not because the baseline sees more detail.** It sees the same
  resized image, and its finest feature map is coarser than the head's grid
  (32-pixel stride against 16-pixel cells). Fine-tuning a whole detector, even
  a small one, adapts to tiny aerial people faster than a frozen model's
  features can be read out by a head.
- **The full-resolution run says where the real gain is.** Feeding the detector
  the 4K frame raised `ap_iou25` on MtErie from 0.213 to 0.779 and `ap_iou50`
  from 0.065 to 0.448. People 4 to 16 pixels across mostly vanish when a frame
  is shrunk to the cache's size, whatever model reads it.

## What it means for the project

The premise was that a frozen foundation model gives a head start when labels
are scarce. On this data, at this resolution, it does not: an ordinary
pretrained detector does better with the same labels. Two consequences:

1. **The bar every later technique has to beat is the fine-tuned detector**,
   not the frozen head. The next epic, Domain adaptation, is measured against
   it.
2. **Resolution is the first lever.** Tiling native-resolution frames, or a
   higher token budget, is likely to matter more than any change to the
   training signal. A test of the frozen head on tiled crops was suggested on
   this issue and not yet run.

## How these numbers were made

- Baseline: 32 runs, `main` at `223dc31`, clean tree, all `completed`, not
  scratch, seed 7 (the same seed and frame selection as the head; the
  comparison refuses runs that read different manifests). Plus one
  full-resolution run from the same commit.
- Head: the 32 #66 runs, `main` at `f93dc3c`. A re-run from a later commit
  reproduced one of them exactly (#68).
- Detector: torchvision `fasterrcnn_mobilenet_v3_large_fpn`, COCO weights,
  anchors 8 to 128 px with five sizes on each of its three feature levels (the
  pretrained proposal head's 15 per location), 300 steps, batch 8, on the CPU.
  MPS diverged or hung.
- About 11 minutes per cache-size run; the 32 took six hours overnight.

```bash
uv run aerial-search train-baseline <rgb|thermal> --fold <site-day> --percent <1|5|10|100> --device cpu
uv run aerial-search train-baseline rgb --fold 210417_MtErie --percent 100 --device cpu --input native
uv run aerial-search compare-table head=outputs/detection-head/siglip2-base-naflex-1024tok \
  baseline=outputs/detection-baseline/fasterrcnn_mobilenet_v3_large_fpn-cache
```

## What the baseline needed to train in the laptop budget

- The CPU, not the GPU: on MPS, Faster R-CNN hung (ResNet-50) or reached a
  loss of about 7e7 (MobileNet).
- MobileNetV3 in place of ResNet-50: ResNet-50 costs about 4 s per step on the
  CPU, which puts the grid at about 8 hours per resolution.
- 300 steps, step selection on 100 evenly spaced validation frames (the head
  uses all of them). Validation was still rising at step 300 in the timing
  run, so the baseline is, if anything, under-trained.
- Anchors scaled down to tiny people, keeping the count the pretrained
  proposal head expects. An earlier version that did not was caught in review
  before any run.

## Limits

- One seed per run, and the spread over folds is as large as some means.
  Six of six cells going the same way at 1% to 10% is the firm part.
- The head may stop training too early at small fractions (#91). The
  matched control in #68 lifted thermal 10% from 0.129 to 0.162, which is still
  well below the baseline's 0.256.
- The baseline selects its step on fewer validation frames than the head.
- One full-resolution run, on the 4K site-day, where resolution helps most.
  The size of the gain on the 2K site-days is unknown.

## Tiled input (#96): design and cost

`train-baseline --input tiled` cuts each source frame into overlapping tiles at
native resolution. The model and recipe are unchanged; the default (`cache`)
and `native` inputs behave as before. No results yet: the grid is not run.

- **Tile size:** the camera's cache input (RGB 672×384, thermal 560×448), so
  the anchors, strides and pretrained weights fit as they do untiled.
- **Overlap:** RGB 160 px, thermal 96 px, so most people sit whole inside at
  least one tile. In the fold manifests RGB boxes are 104×146 px at the 90th
  percentile (up to 972 px) and thermal 50×76 px (up to 190×431). No overlap
  can hold the largest boxes in a 384-px-high tile; those are seen cut. A 1080p
  RGB frame gives 20 tiles, a 4K frame 72, a thermal frame 4.
- **Training, cut boxes:** a box is kept, clipped to the tile, when at least
  half its area is inside (`--min-visible 0.5`); less than that is dropped from
  that tile, so a sliver never teaches the model to fire on a person's edge.
- **Training, empty tiles:** every tile with a person, plus random empty tiles
  at 1 per person tile (`--empty-ratio 1.0`), chosen once from the seed. Most
  tiles are empty (Baker RGB 10%: 580 frames, 14,720 tiles, 2,181 with a
  person); keeping all would swamp the signal. Steps stay 300 at batch 8.
- **Scoring:** each tile's boxes move to source pixels; class-agnostic NMS at
  IoU 0.5 over the stitched frame merges a person seen in an overlap (the
  cut view of a person overlaps the whole view above 0.5 when at least about
  half is visible, which is the training cut-off); the top 300 per frame are
  kept (`--frame-top-k`); then the existing `evaluate_detections` scores whole
  frames on the same folds and test sets.
- **Cost (laptop CPU, scratch, untrained-then-briefly-trained model):** about
  3.4 s per training step (3.4 RGB, 4.0 thermal) and 0.11 s per tile at test
  time: 2.2 s per 1080p RGB frame, 5.0 s per 4K frame, 1.4 s per thermal
  frame. Tiles are cropped on the fly, so extra disk is nil. Per cell:
  training about 17 to 20 min, validation 7 to 11 min, test 1 to 2.5 h. The
  grid (rgb and thermal, 4 folds, 10% and 100%) is about 26 h, against 6 h for
  the 32 untiled runs. The test pass is most of it (RGB FHL alone, 4,154
  frames, is about 2.5 h), so no cell fits the 30-minute rule.
