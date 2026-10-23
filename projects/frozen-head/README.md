# frozen-head

The second approach: a frozen vision foundation model (SigLIP 2) turns each
image into features once, and a small person-centre head is trained on those
cached features with few labels. Scored by the shared evaluation in
`shared/` against the fine-tuned detector in `projects/finetuned-detector/`.

## Goal

Find out how far frozen general-purpose features carry person detection in
aerial RGB and thermal frames when labels are scarce.

## Approach

`cache-features` runs the frozen SigLIP 2 NaFlex backbone once per image at
a 1,024-token budget and caches pooled (2×2) fp16 patch features, keyed by
model, weight checksum, token budget, pooling and dtype (#64, #65,
[docs/forward-pass-cost.md](../../docs/forward-pass-cost.md)). `train-head`
trains a CenterNet-style head (Zhou, Wang and Krähenbühl, *Objects as
Points*, 2019) on a grid four times finer than the feature grid
([docs/detection-head-design.md](../../docs/detection-head-design.md)).
Thermal frames reach the backbone with the grey channel copied three times;
equalising them or a learned input stem did not help (#68,
[docs/thermal-input-ablation.md](../../docs/thermal-input-ablation.md)).

## Expected results

Written before any run ([docs/detection-head-design.md](../../docs/detection-head-design.md)):
RGB poor, `ap_iou25` around 0.05 to 0.2 at 100% labels, `ap_iou50` well
below it; thermal clearly better, `ap_iou25` perhaps 0.2 to 0.5, with
`ap_iou50` closer to it; a large gap between the 1% and 100% rows for both.
Thermal landed inside its range and RGB just above, but thermal `ap_iou50`
did not sit close to `ap_iou25`
([docs/detection-head-results.md](../../docs/detection-head-results.md)).

## Run

```bash
uv sync --dev
uv run aerial-search cache-features siglip2-base-naflex --camera rgb --part 1/2
uv run aerial-search train-head rgb --fold 220109_Baker --percent 10
uv run aerial-search head-table outputs/detection-head/siglip2-base-naflex-1024tok
uv run pytest projects/frozen-head
```

## Results

Test `ap_iou25` / `ap_iou50` per fold, then the mean ± sd over the four
folds. 32 runs from `main` at `f93dc3c`, seed 7.

| Camera | Labels | MtErie | Carnation | FHL | Baker | `ap_iou25` | `ap_iou50` |
|---|---|---|---|---|---|---|---|
| RGB | 1% | 0.031 / 0.001 | 0.079 / 0.003 | 0.026 / 0.002 | 0.038 / 0.001 | 0.044 ± 0.024 | 0.001 ± 0.001 |
| RGB | 5% | 0.062 / 0.001 | 0.089 / 0.004 | 0.082 / 0.009 | 0.144 / 0.004 | 0.094 ± 0.035 | 0.004 ± 0.003 |
| RGB | 10% | 0.077 / 0.002 | 0.115 / 0.008 | 0.045 / 0.002 | 0.216 / 0.012 | 0.113 ± 0.075 | 0.006 ± 0.005 |
| RGB | 100% | 0.226 / 0.030 | 0.150 / 0.026 | 0.086 / 0.007 | 0.540 / 0.163 | 0.250 ± 0.201 | 0.056 ± 0.072 |
| Thermal | 1% | 0.063 / 0.001 | 0.000 / 0.000 | 0.028 / 0.003 | 0.168 / 0.002 | 0.065 ± 0.073 | 0.001 ± 0.001 |
| Thermal | 5% | 0.177 / 0.004 | 0.001 / 0.000 | 0.076 / 0.012 | 0.276 / 0.012 | 0.132 ± 0.120 | 0.007 ± 0.006 |
| Thermal | 10% | 0.229 / 0.010 | 0.001 / 0.000 | 0.121 / 0.011 | 0.165 / 0.008 | 0.129 ± 0.096 | 0.007 ± 0.005 |
| Thermal | 100% | 0.313 / 0.057 | 0.003 / 0.000 | 0.202 / 0.030 | 0.692 / 0.247 | 0.302 ± 0.290 | 0.083 ± 0.111 |

The fine-tuned detector beats this head at 1%, 5% and 10% on both cameras.

Source: [docs/supervised-baseline-results.md](../../docs/supervised-baseline-results.md)
(the "head" rows; also in [docs/detection-head-results.md](../../docs/detection-head-results.md)).

## Status (2026-10-23)

- Done: the feature cache (#65), the detection head (#66), the thermal input
  ablation (#68).
- Doing: nothing.
- Next: adapt the backbone to aerial imagery (the Domain adaptation epic).
