# finetuned-detector

The bar every other approach has to beat: an ordinary COCO-pretrained
torchvision detector (Faster R-CNN) fine-tuned on the same few labelled
frames as the others, scored by the shared evaluation in `shared/`.

## Goal

A supervised detector, trained per fold, camera, label fraction and seed,
whose test scores are the baseline for every approach in `projects/`.

## Status (2026-10-23)

- Done: fine-tuning at the cache's input size, native size or on tiles
  (#67, #96); checkpoints and resumable test predictions (#112); the SageMaker
  spot launcher (#113). Results: `docs/supervised-baseline-results.md`.
- Next: the tiled runs, on SageMaker.

## Layout

- `src/finetuned_detector/`: the code.
- `tests/`: tests for it.

## Run

```bash
uv sync --dev
uv run aerial-search train-baseline rgb --percent 10 --fold 220109_Baker --scratch
uv run aerial-search sagemaker-train rgb --percent 10 --fold 220109_Baker \
  --input tiled --dry-run
uv run pytest projects/finetuned-detector
```
