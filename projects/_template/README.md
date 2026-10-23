# my-approach

<What this approach does to find people in aerial RGB and thermal frames, in
two sentences.>

## Goal

<What is true when this approach is done, and what the owner will
understand afterwards.>

## Approach

<The technique, its source (paper or library, with a link), and why it might
beat the bar in `projects/finetuned-detector/` when labels are scarce.>

## Expected results

<Written down before the first run, and never edited after it: the
`ap_iou25` and `ap_iou50` you expect per camera and label fraction, and the
numbers that would count as success or failure against the bar.>

## Run

```bash
uv sync --dev
uv run aerial-search train-my-approach rgb --percent 10 --fold 220109_Baker --scratch
uv run pytest projects/my-approach
```

## Results

Test `ap_iou25` / `ap_iou50` per fold (each fold's test set is one site-day),
then the mean ± sample sd over the folds. Only completed runs from a clean
`main` commit; give the commit and the command that made them.

| Camera | Labels | MtErie | Carnation | FHL | Baker | `ap_iou25` | `ap_iou50` |
|---|---|---|---|---|---|---|---|
| rgb | 10% | not yet measured | | | | | |

Source: <the docs/ file that holds these numbers and how they were made>.

## Status (YYYY-MM-DD)

- Done: <...>
- Doing: <...>
- Next: <...>
