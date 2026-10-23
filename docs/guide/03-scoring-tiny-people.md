# 3. Scoring tiny people

*Last updated 2026-10-01 from issue [#4]. The label-completeness caveat comes from the
open issue [#37].*

## What we set out to do

Build one scorer that every later experiment shares, before any model exists.
The earlier plan quoted mAP for a model that predicts grid cells, and mAP is
not defined for that ([#4]). People in this footage are a few pixels across,
so the choice of metric decides what a result means.

## What we found

**Strict overlap punishes errors a searcher does not care about.** A detection
counts as a hit when its overlap with a person (intersection over union, IoU)
passes a threshold. For a square person of side s, and a detection of the same
size shifted d pixels right and down, the overlap is (s - d)^2. From that
([evaluation review][metric-review]):

| Person | Shift | IoU | Hit at 0.5? | Hit at 0.25? |
|---|---|---|---|---|
| 12 x 12 | 2 px | 0.53 | yes | yes |
| 10 x 10 | 2 px | 0.47 | no | yes |
| 8 x 8 | 2 px | 0.39 | no | yes |
| 8 x 8 | 3 px | 0.24 | no | no |

So a 2 pixel error fails IoU 0.5 on people 10 pixels or smaller, and a 3
pixel error fails it on people 16 pixels or smaller ([evaluation review][metric-review]).
At 0.25 a detection must still overlap the person substantially: the last row
fails. AI-TOD, an aerial tiny-object set, has a mean object size of 12.8
pixels ([evaluation review][metric-review], citing Wang et al., 2021).

**What the scorer reports** ([evaluation review][metric-review]):

| Metric | Role |
|---|---|
| `ap_iou25`, average precision at IoU 0.25 | Primary |
| `recall_at_fppi`, best recall at 0.01, 0.1 and 1 false alarms per image | The search team's number |
| `ap_iou50` and `ap_coco` (mean over IoU 0.50 to 0.95) | For comparison with other work |

Everything is given overall and by modality, flight and person size. The size
buckets are on the square root of box area: `very_tiny` [0, 8), `tiny` [8, 16),
`small` [16, 32), `medium_plus` [32, inf).

**The first version flattered its own numbers, in two ways** ([#4 handoff][i4-handoff]).

1. Per-size recall ignored false alarms of other sizes. The reviewer's case: a
   6 x 6 person found at score 0.5 and a 20 x 20 false alarm at 0.9 gave
   very-tiny recall 1.0 at 0.1 false alarms per image. The right answer is 0.0
   ([evaluation review][metric-review]).
2. When scores tied, the result depended on the order of the input.

The lead's own hand-worked case caught neither ([#4 handoff][i4-handoff]).
Two reviews by the `reviewer` agent did. After the fixes, the second review
ran 600 random cases against the stock library, 3,000 cases over every
ordering of tied scores, and 18 deliberate changes to the code. The tests now
catch all 18 ([#4 handoff][i4-handoff]).

**Ties can only lower a score.** False alarms rank ahead of hits that share
their score. Against stock pycocotools, with no tied scores `ap_iou50` matched
in all 288 random cases. With ties, 143 of 282 cases differed, and ours was
never higher. The largest gap was 0.67 ([evaluation review][metric-review]).

## Decisions and why

- **AP at IoU 0.25 as primary, with `ap_iou50` always beside it.** Both are
  published wherever `ap_iou25` is, a rule in `CLAUDE.md`, so a reader can see
  how much the lower threshold contributes ([#4 handoff][i4-handoff]).
- **Recall at fixed false alarms per image.** It is what a search team asks:
  how likely is a person flagged, and what false alarms must I accept?
- **pycocotools for matching, with two corrections.** Its recall levels are off
  by a float step at ten points, and its size ranges overlap at both ends. The
  corrections move `ap_iou50` by at most 0.0087 over 300 random cases
  ([evaluation review][metric-review]).
- **No cap on detections per image.** pycocotools silently keeps 100.
- **Per-size recall counts false alarms of every size.** A searcher sees all of
  them. Per-size AP keeps the COCO rule, so it can be compared with other work.
- **Not used:** normalised Wasserstein distance (a loss and assignment tool,
  not an evaluation measure) and centre-distance matching (a new free
  parameter no benchmark uses) ([evaluation review][metric-review]).

## How to reproduce

```bash
uv sync --dev
uv run pytest shared/aerial_search/tests/test_detection_evaluation.py
```

On 2026-10-01 the output was `27 passed in 1.05s`. The tests hold the
hand-worked answers: perfect predictions, none, images with no people, and
duplicate detections. A person of 10 x 10 and a detection shifted 2 pixels
shows the table's second row:

```python
from aerial_search.evaluation.detection import (
    GroundTruth,
    Prediction,
    evaluate_detections,
)

person = (0.0, 0.0, 10.0, 10.0)
shifted = (2.0, 2.0, 12.0, 12.0)
truths = [GroundTruth("img1", [person], "rgb", "demo")]
preds = [Prediction("img1", [shifted], [0.9])]
m = evaluate_detections(truths, preds).overall
print(m.ap_iou25, m.ap_iou50)  # 1.0 0.0
```

Boxes are `(x1, y1, x2, y2)` in pixels, far edge exclusive. `report.write_json(run_dir)`
writes `detection_metrics.json` into a run directory.

## What we would do differently

- **Do not trust one hand-worked case.** The lead's caught neither flaw. Two
  reviews did, and the tests now pin both ([#4 handoff][i4-handoff]).
- **Read the key paper.** The implementer could not open the TinyPerson paper
  and relied on search results that quote it. The case for 0.25 does not depend
  on it ([#4 handoff][i4-handoff]).
- **Left alone for now** ([#4 handoff][i4-handoff]): tied detections within one
  image are matched in coordinate order, which is fixed but not the least
  favourable; the size breakdown is overall only; the by-flight breakdown mixes
  RGB and thermal images of one flight.

## What the score cannot say

At 0.01 false alarms per image the estimate rests on few false alarms: on 500
test images it is decided by the first 5 ([evaluation review][metric-review]).
Recall is also measured against an incomplete list of people. A first-pass
estimate puts the unboxed share at 1.0% of people in RGB (0.3 to 3.5) and 5.7%
in thermal (3.1 to 10.1). All 10 missed people in the thermal sample come
from one clip ([label review][label-review]). That review is provisional until the owner
checks 21 frames.

## Next

[Chapter 4](04-splitting-by-site-day.md): what the scorer is run on.

[#4]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/4
[#37]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/37
[i4-handoff]: https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/4#issuecomment-5892972875
[metric-review]: ../evaluation-metric-review.md
[label-review]: ../label-completeness-review.md
