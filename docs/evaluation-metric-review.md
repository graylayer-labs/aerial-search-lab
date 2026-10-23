# Evaluation metric review

How detectors are scored in this project, and why. The code is
`shared/aerial_search/src/aerial_search/evaluation/detection.py`; the tests with hand-worked
answers are `shared/aerial_search/tests/test_detection_evaluation.py`. Issue:
[#4](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/4).
Review: [#34](https://github.com/graylayer-labs/ssl-aerial-person-detection/pull/34).

## What is reported

| Metric | Definition | Role |
|---|---|---|
| `ap_iou25` | Average precision where a detection counts as a hit if its IoU with an unmatched person is at least 0.25. | **Primary** |
| `recall_at_fppi` | The best recall reachable at a score cutoff whose false alarms per image are at most 0.01, 0.1, or 1, with hits judged at IoU 0.25. | The search team's number |
| `ap_iou50` | The same average precision with the IoU threshold at 0.5. | For comparison with other work |
| `ap_coco` | The mean of average precision over IoU 0.50, 0.55, ..., 0.95. | For comparison with other work |
| `n_predictions_tied` | How many detections share their score with another detection in the group. | Shows when ties matter |

Every metric is given overall, by modality, by flight, and by person size.
A group with no people has undefined AP and recall, written as `null`.
Every published table or figure that shows `ap_iou25` shows `ap_iou50` beside
it (a rule in `CLAUDE.md`), so a reader can see how much the lower threshold
contributes.

Average precision is COCO's 101-point interpolation: the mean, over recall
levels 0.00, 0.01, ..., 1.00, of the highest precision reached at that recall
or above, with 0 for levels never reached.

## Why AP at IoU 0.25 is the primary metric

People in aerial search footage are a few pixels across. AI-TOD, an aerial
tiny-object dataset, has a mean object size of 12.8 pixels ([Wang et al.,
2021](https://arxiv.org/abs/2110.13389)). At that size IoU punishes errors that
do not matter to a searcher.

For a square person of side s and a detection of the same size shifted d
pixels right and d pixels down, the overlap is (s - d)² and the union is
2s² - (s - d)². Worked cases:

| Person | Shift d | IoU | Hit at 0.5? | Hit at 0.25? |
|---|---|---|---|---|
| 12 x 12 | 2 | 100 / 188 = 0.53 | yes | yes |
| 12 x 12 | 3 | 81 / 207 = 0.39 | no | yes |
| 10 x 10 | 2 | 64 / 136 = 0.47 | no | yes |
| 8 x 8 | 2 | 36 / 92 = 0.39 | no | yes |
| 8 x 8 | 3 | 25 / 103 = 0.24 | no | no |
| 6 x 6 | 2 | 16 / 56 = 0.29 | no | yes |

Solving IoU >= t for s gives the sizes where the claim holds:

- IoU >= 0.5 needs 1.5 (s - d)² >= s², so s >= d √1.5 / (√1.5 - 1) = 5.45 d.
  A 2-pixel diagonal error fails 0.5 on people 10 pixels or smaller; a
  3-pixel error fails it on people 16 pixels or smaller.
- IoU >= 0.25 needs 2.5 (s - d)² >= s², so s >= d √2.5 / (√2.5 - 1) = 2.72 d.
  A 2-pixel error passes 0.25 on people 6 pixels or larger; a 3-pixel error
  on people 9 pixels or larger.

So an error of two pixels on a person 10 pixels or smaller, or three pixels
on a person 16 pixels or smaller, still puts a searcher's eyes on the right
spot, and is both a miss and a false alarm at 0.5. At 0.25 a detection must still overlap the person substantially: the
8 x 8 row with a 3-pixel shift fails it.

The NWD paper makes the same point with a 1-pixel and a 4-pixel diagonal
shift. For a 6 x 6 object IoU falls from 25/47 = 0.53 to 4/68 = 0.06; for a
36 x 36 object it falls only from 1225/1367 = 0.90 to 1024/1568 = 0.65. (The
fractions are worked from the formula above and match the paper's figures.)

TinyPerson, the closest benchmark in purpose, evaluates at IoU 0.25 as well
as 0.5. **I could not open the TinyPerson paper**: both the arXiv PDF and page
came back unreadable to my tools. What I relied on is search-result text that
quotes it: that the benchmark uses IoU 0.25 and 0.5 (and 0.75), and that it
adds 0.25 because many tiny-person applications, such as shipwreck search and
rescue, care more about finding people than locating them precisely
([Yu et al., WACV 2020](https://arxiv.org/abs/1912.10664)). The case for 0.25
above does not depend on it.

## Why recall at false alarms per image

A search team asks: if a person is in the frame, how likely is the system to
flag them, and how many false alarms per image must I accept for that? AP does
not answer this. Recall at fixed false alarms per image (FPPI) does. It is the
standard measure in pedestrian detection, where the Caltech benchmark
summarises miss rate over FPPI from 0.01 to 1 ([Dollár et al., TPAMI
2012](https://doi.org/10.1109/TPAMI.2011.155)). The three points reported
span that range.

Definition, as implemented:

- Each distinct detection score in the group is a cutoff.
- At a cutoff c, recall = hits scoring >= c / people, and FPPI = false alarms
  scoring >= c / images. Images count whether or not they contain people; a
  test pins this denominator.
- Recall at FPPI f is the highest recall over cutoffs with FPPI <= f. Keeping
  no detections (recall 0, FPPI 0) is always allowed.

Detections with equal scores enter at one cutoff, because no threshold can
keep one and drop the other. For the overall figure and the breakdowns by
modality and flight, hits and false alarms come from the same pycocotools
matching that produces AP. Per size they come from two matchings, described
next.

**Per size, false alarms of every size count.** In a size bucket, hits are
the bucket's people found, but false alarms are all false alarms in the group
at the same cutoff, whatever the size of their box. A searcher sees every
false alarm. An earlier version counted only false alarms of the bucket's own
size, which made per-size recall look better than it is (reviewer's case: a
6 x 6 person found at 0.5 and a 20 x 20 false alarm at 0.9 gave very-tiny
recall 1.0 at FPPI 0.1 instead of 0.0).

Per-size AP keeps the COCO rule instead, so it can be compared with other
work: an unmatched detection of another size is ignored in the bucket. The
two metrics therefore treat false alarms of other sizes differently, and on
purpose.

At FPPI 0.01 the estimate rests on few false alarms: on 500 test images it is
decided by the first 5. Treat it as noisy on small splits.

## Matching rules

pycocotools does the per-image matching. The tests pin each rule.

- Within an image, detections are taken highest score first. Each takes the
  unmatched person with the highest IoU, provided that IoU is at least the
  threshold. IoU exactly equal to the threshold counts.
- A person is matched once. A second detection of the same person is a false
  alarm (a duplicate), unless another unmatched person overlaps it enough.
- When a detection overlaps two people, it takes the one with the higher IoU,
  not the first listed. Equal IoU goes to the person later in box coordinate
  order.
- Every detection on an image with no people is a false alarm.
- There is no cap on detections per image. pycocotools caps at 100 by default
  and silently drops the rest; the scorer sets the cap to the largest count
  present. On any image with more than 100 detections, results therefore
  differ from stock pycocotools with its default cap, and the difference has
  no bound: it depends on what the dropped detections were.

## Ties in score

No reported number depends on the order of the input. A test shuffles the
ground-truth list, the prediction list, and the boxes within each image eight
ways and checks the whole report is identical. Ties are common in practice: a
float32 sigmoid returns exactly 1.0 for any logit above about 17.

- Scores are converted to float64 on the way in.
- **Within an image**, detections with equal scores are matched in box
  coordinate order (x1, y1, x2, y2), and people are listed in the same order.
  This order is fixed but arbitrary: it is neither the optimistic nor the
  pessimistic matching. Finding the pessimistic matching means searching over
  orderings, which was not done.
- **On the precision-recall curve**, false alarms with the same score as hits
  enter first. That is the order that gives the lower AP. Stock pycocotools
  orders them by image, then by input order, so it can give a higher AP.
- **Recall at FPPI** treats equal scores as one cutoff (above).
- In a size bucket, `n_predictions` and `n_predictions_tied` count detections
  by the size of the predicted box, while the bucket's AP is decided by the
  size of the person matched. The counts and the AP therefore do not describe
  exactly the same detections.
- `n_predictions_tied` reports how many detections share a score, so a reader
  can see when ties could matter.

Difference from stock pycocotools. AP is computed here from pycocotools'
per-image matches rather than by its `accumulate()`, because `accumulate()`
breaks ties by input order. Measured on random cases (up to 7 images, up to
4 people each, with the same annotation order given to both):

- Without tied scores, `ap_iou50` was identical to stock `accumulate()` on the
  exact recall grid in all 288 cases.
- With scores rounded to one decimal, so ties are frequent, 143 of 282 cases
  differed. Ours was never higher; the largest gap was 0.67, on small cases
  where one tie group held most of the detections. The gap is bounded only by
  how many hits share a score with false alarms.

## Input format

- `GroundTruth(image_id, boxes, modality, flight)`, one per image, including
  images with no people. This list defines the images evaluated.
- `Prediction(image_id, boxes, scores)`, one per image at most. A missing image
  has no detections. A prediction for an image not in the ground truth is an
  error.
- Boxes are `(x1, y1, x2, y2)` in pixels, origin top-left, continuous
  coordinates with the far edge exclusive: a box over pixel columns 5 to 9 is
  `x1=5, x2=10`. Labels stored as inclusive pixel indices need 1 added to `x2`
  and `y2`. Boxes with zero or negative width or height are rejected.

The format depends on nothing from the model code, which will be rebuilt.

## Size buckets

On the square root of box area, half-open, following AI-TOD's edges:
`very_tiny` [0, 8), `tiny` [8, 16), `small` [16, 32), `medium_plus` [32, inf).
COCO's buckets would put nearly every person in "small". A person's bucket
comes from the ground-truth box. For AP, people of other sizes are ignored in
a bucket, as are unmatched detections of other sizes (the pycocotools rule).

**Per-size figures do not add up to the overall figure.** Because each bucket
ignores people of other sizes, one detection can be a hit in two buckets. The
reviewer's case, pinned by a test: a 10 x 10 person with a 6 x 6 person inside
it, and one detection equal to the larger. Overall the detection matches the
larger person, recall is 1/2, and AP is 51/101. In `tiny` it matches the
larger person and in `very_tiny` the smaller one (IoU 36/100 = 0.36), so both
buckets report AP 1.0. This is standard COCO behaviour.

## Library choice

pycocotools, for IoU and the per-image matching.

- It accepts custom IoU thresholds, area ranges and detection caps, and
  exposes its per-image matches (`evalImgs`), from which AP and recall at
  FPPI are computed.
- It is a single small package with wheels for our platforms.

Considered and not used:

- **torchmetrics `MeanAveragePrecision`**: wraps pycocotools or
  faster-coco-eval, so it adds a dependency without removing one. Its area
  ranges are fixed to COCO's, its detection caps are a list of exactly three,
  and it exposes no per-image matches, so FPPI could not be computed from it
  ([source](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py)).
- **Writing the matching from scratch**: a second implementation of a
  standard rule is where silent disagreements with published numbers come
  from. The AP accumulation is short, and was checked against stock
  `accumulate()` as described above.

Two pycocotools behaviours are corrected, each pinned by a test that fails
without the correction:

- Its recall levels come from `np.linspace`, which puts ten of them one float
  step above k/100 (0.35, 0.41, 0.47, 0.57, 0.69, 0.70, 0.82, 0.83, 0.94,
  0.95). A recall of exactly 0.7 then misses level 0.70. The scorer uses exact
  k/100. This can move AP by at most one level's precision divided by 101,
  compared with stock pycocotools. For both corrections together the reviewer
  measured at most 0.0087 on `ap_iou50` over 300 random cases.
- Its area ranges are closed at both ends, so an 8 x 8 person would fall in
  two buckets. The scorer steps each upper edge down by one float step, making
  buckets half-open.

## Considered and not used as the primary metric

- **AP at IoU 0.5 or COCO 0.5:0.95 as primary.** Too strict for people a few
  pixels across; see the table above. Both are still reported.
- **Normalised Wasserstein distance (NWD) as the matching measure.** Designed
  for label assignment, NMS and the loss, not evaluation. It needs a constant C
  that the authors tie to the dataset's mean object size, so scores would not
  compare across datasets ([Wang et al., 2021](https://arxiv.org/abs/2110.13389)).
  AI-TOD itself still evaluates with IoU-based AP.
- **Centre-distance matching.** Scale-free at a fixed pixel radius, but the
  radius is a new free parameter and no aerial person benchmark uses it, so
  results could not be compared.

## Judgement calls that could move a headline number

- The 0.25 threshold. At 0.5, AP on tiny people will be much lower; hence the
  rule that `ap_iou50` is always shown beside it.
- The FPPI points. Denominators include empty images, so a test split with
  more empty frames shows lower FPPI for the same detector.
- 101-point interpolation rather than the all-point area under the curve. A
  detector with precision 1 up to recall 0.5 scores 51/101, not 0.5.
- The tie rules. The curve rule is pessimistic; the within-image matching
  order is arbitrary. Check `n_predictions_tied` before trusting a small
  difference between two models.

## Sources

- Wang, Xu, Yang, Yu. A Normalized Gaussian Wasserstein Distance for Tiny
  Object Detection. 2021. <https://arxiv.org/abs/2110.13389>
- Yu, Gong, Jiang, Ye, Han. Scale Match for Tiny Person Detection. WACV 2020.
  <https://arxiv.org/abs/1912.10664> (not read directly; see above)
- Dollár, Wojek, Schiele, Perona. Pedestrian Detection: An Evaluation of the
  State of the Art. TPAMI 2012. <https://doi.org/10.1109/TPAMI.2011.155>
- pycocotools `cocoeval.py`. <https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py>
- torchmetrics `mean_ap.py`. <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py>
