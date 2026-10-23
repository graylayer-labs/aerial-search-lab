# frozen-head

The second approach: a frozen vision foundation model (SigLIP 2) turns each
image into features once, and a small person-centre head is trained on those
cached features with few labels. Scored by the shared evaluation in
`shared/` against the fine-tuned detector in `projects/finetuned-detector/`.

## Goal

Find out how far frozen general-purpose features carry person detection in
aerial RGB and thermal frames when labels are scarce.

## Status (2026-10-23)

- Done: the feature cache (#65), the detection head (#66), the thermal input
  ablation (#68). The fine-tuned detector beats the head wherever labels are
  scarce: `docs/detection-head-results.md`, `docs/supervised-baseline-results.md`.
- Next: adapt the features to aerial imagery (domain adaptation epic).

## Layout

- `src/frozen_head/`: the backbone, the feature cache, the head and its
  experiments.
- `tests/`: tests for it.

## Run

```bash
uv sync --dev
uv run aerial-search cache-features siglip2-base-naflex --camera rgb --part 1/2
uv run aerial-search train-head rgb --fold 220109_Baker --percent 10
uv run pytest projects/frozen-head
```
