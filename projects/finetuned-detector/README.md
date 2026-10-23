# finetuned-detector

The bar every other approach has to beat: an ordinary COCO-pretrained
torchvision detector fine-tuned on the same few labelled frames as the
others, scored by the shared evaluation in `shared/`.

## Goal

A supervised detector, trained per fold, camera, label fraction and seed,
whose test scores are the baseline for every approach in `projects/`.

## Approach

torchvision `fasterrcnn_mobilenet_v3_large_fpn` with COCO weights, anchors
8 to 128 px, 300 steps at batch 8 on the CPU, trained on the frames
`aerial_search.data.records.select_records` picks, the same frames as the
frozen head ([#67](../../docs/supervised-baseline-results.md)). It reads the
frames at the cache's input size (the fair row against the frozen head), at
native size, or as overlapping native-resolution tiles stitched back to
source pixels ([#96](../../docs/supervised-baseline-results.md)).
Checkpoints and test predictions are saved as it goes, so a stopped run
resumes (#112), and `sagemaker-train` runs it as a SageMaker spot job (#113).

## Expected results

Not written down before the runs (#67). Later approaches write theirs here
before their first run.

## Run

```bash
uv sync --dev
uv run aerial-search train-baseline rgb --fold 210417_MtErie --percent 10 --device cpu
uv run aerial-search sagemaker-train rgb --percent 10 --fold 220109_Baker \
  --input tiled --dry-run
uv run pytest projects/finetuned-detector
```

## Results

At the cache's input size (672×384 RGB, 560×448 thermal): test `ap_iou25` /
`ap_iou50` per fold, then the mean ± sd over the four folds. 32 runs from
`main` at `223dc31`, seed 7.

| Camera | Labels | MtErie | Carnation | FHL | Baker | `ap_iou25` | `ap_iou50` |
|---|---|---|---|---|---|---|---|
| RGB | 1% | 0.016 / 0.000 | 0.101 / 0.011 | 0.049 / 0.018 | 0.092 / 0.007 | 0.064 ± 0.039 | 0.009 ± 0.007 |
| RGB | 5% | 0.152 / 0.039 | 0.157 / 0.049 | 0.089 / 0.035 | 0.209 / 0.079 | 0.152 ± 0.049 | 0.051 ± 0.020 |
| RGB | 10% | 0.167 / 0.036 | 0.190 / 0.052 | 0.102 / 0.036 | 0.192 / 0.058 | 0.163 ± 0.042 | 0.046 ± 0.011 |
| RGB | 100% | 0.213 / 0.065 | 0.201 / 0.061 | 0.099 / 0.038 | 0.302 / 0.077 | 0.204 ± 0.083 | 0.060 ± 0.016 |
| Thermal | 1% | 0.130 / 0.048 | 0.002 / 0.000 | 0.073 / 0.026 | 0.406 / 0.214 | 0.153 ± 0.177 | 0.072 ± 0.097 |
| Thermal | 5% | 0.256 / 0.107 | 0.009 / 0.001 | 0.162 / 0.062 | 0.482 / 0.279 | 0.228 ± 0.198 | 0.112 ± 0.119 |
| Thermal | 10% | 0.311 / 0.129 | 0.002 / 0.000 | 0.186 / 0.078 | 0.527 / 0.341 | 0.256 ± 0.220 | 0.137 ± 0.146 |
| Thermal | 100% | 0.368 / 0.180 | 0.007 / 0.000 | 0.220 / 0.114 | 0.580 / 0.410 | 0.294 ± 0.242 | 0.176 ± 0.173 |

One run at full resolution (RGB, MtErie, 100%) scored 0.779 / 0.448.
Tiled input: not yet measured.

Source: [docs/supervised-baseline-results.md](../../docs/supervised-baseline-results.md).

## Status (2026-10-23)

- Done: fine-tuning at the cache's input size, native size or on tiles
  (#67, #96); checkpoints and resumable test predictions (#112); the
  SageMaker spot launcher (#113).
- Doing: nothing.
- Next: the tiled runs, on SageMaker.
