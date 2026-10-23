# Site-day folds: how WiSARD is split, and what it can support

Issue [#3](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/3).
Code: `shared/aerial_search/src/aerial_search/data/folds.py`. Evidence for the gap:
`tools/frame_similarity_lags.py`.

## Decision

The unit that never crosses a split is the **site-day**: every clip recorded
at one site on one date. Neighbouring frames are near-duplicates, and clips
from one site-day share terrain, light, weather, and often the same people.

- **Leave one site-day out.** Each labelled site-day is the test set of one
  fold: MtErie, Carnation, FHL, Baker. Results are reported per fold and as
  the mean and spread over the four folds. The claim they support is about
  an unseen site-day, not an unseen site: `data/raw/wisard-full` also holds
  earlier, unused flights at Carnation and Baker. Widening the unlabelled
  pool to those flights would need the folds to exclude by site, not
  site-day.
- **Hannegan has no labels.** It is never a test set, and it is unlabelled
  data in every fold.
- **Three label views**, one set of folds each: `paired` (labelled in both
  cameras), `rgb`, `thermal`. The test site-day of a fold is the same in all
  three. A view's test set holds every frame of the test site-day labelled in
  that view.
- **Validation** comes from the training site-days. A site-day with two or
  more clips labelled in the view gives its smallest labelled clip, whole. A
  site-day with one labelled clip gives the last 15% of that clip, and the 250
  frames before it are used nowhere.
- **Unlabelled pool** (`unlabelled.jsonl`, one per fold): every pair not from
  the test site-day, minus every frame that any view's validation holds or
  that any view's gap removes. So self-supervised pretraining sees neither
  the test site nor the validation frames. One pool serves all three views.
- **Label overrides**: a clip's empty-label frames can be treated as
  unlabelled where the empty label is not trusted (below). They are in no
  manifest of that camera, in any fold, and stay in the unlabelled pool.
- **Label fractions**: 1%, 5%, 10%, 100% of each view's training labels,
  nested, seeded (seed 7), made of blocks of 10 consecutive labelled
  records of one clip.

A site-day is derived from the directory name, never typed by hand:
`site_day("210924_FHL_Enterprise_VIS_0126")` is `210924_FHL`. A test checks
the derivation on all 17 collections (five site-days: MtErie 3 clips,
Carnation 2, Hannegan 2, FHL 9, Baker 1), and the builder refuses a record
whose image directory names another site-day than its clip.

## Label overrides

`LABEL_OVERRIDES` in `shared/aerial_search/src/aerial_search/data/folds.py` lists (clip, camera,
rule, reason) entries and `folds.json` records each with its frame count. The
rule `empty_label` names the frames whose label file exists and holds no box.
The first entry is the thermal view of `210924_FHL_Enterprise_0403` (thermal
directory `210924_FHL_Enterprise_IR_0404`), issue
[#53](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/53).

- **Why.** [#49](https://github.com/graylayer-labs/ssl-aerial-person-detection/issues/49)
  found labelling stops about 215 frames in. In a 60-frame sample, 30 of the
  35 empty-label frames judged with certainty held an unboxed person (85.7%,
  95% interval 70.6 to 93.7); scaled to the clip, roughly 1,030 of its 1,201
  empty-label frames (850 to 1,125). Left in, they teach a thermal detector
  that visible people are background and make the FHL thermal test set charge
  false alarms for finding real people. Evidence:
  `docs/label-completeness-review.md`, "Thermal clip FHL_0403".
- **Effect.** The 1,201 frames leave every thermal train, validation, and
  test manifest and stay in the pool of the folds where FHL is a training
  site (MtErie, Carnation, Baker); the FHL fold's pool already excludes the
  site-day. The pool is built from pairs, so it is unchanged. The clip's
  215 or so boxed frames stay. No label file is changed, and the rgb and
  paired views are untouched (the paired view still holds this clip's pairs,
  whose thermal side is empty; not addressed here).
- **Validation is unchanged.** The override is applied after the split, so
  validation still comes from the same clips; with the override this clip
  would otherwise shrink to 216 frames, fewer than clip 0134's 273, and
  become the validation clip.
- **Check.** `check-folds` finds the frames again from the source manifest
  (clip directory, empty label), and fails if one is in any thermal manifest,
  if one is missing from the pool of a fold that trains on FHL, or if
  `folds.json` does not record the override.

Thermal frames per fold, before and after (boxes do not change, as the
removed frames hold none):

| Fold | Train 100%, before | after | Test, before | after |
|---|---|---|---|---|
| MtErie | 7,191 | 5,990 | 708 | 708 |
| Carnation | 7,286 | 6,085 | 740 | 740 |
| FHL | 2,580 | 2,580 | 5,420 | 4,219 |
| Baker | 6,124 | 4,923 | 2,180 | 2,180 |

Validation frames are unchanged in all four folds.

**The thermal label-fraction subsets are redrawn.** Subsets are drawn after
the override, and the draw shuffles blocks per site-day, so removing this
clip's blocks reorders the FHL blocks. The 1%, 5%, and 10% thermal subsets of
the MtErie, Carnation, and Baker folds therefore hold different frames and
different box counts from a build without the override, not merely fewer.
The 100% training sets and the test sets lose exactly the 1,201 frames and no
boxes. No thermal result was produced before the override, so nothing is made
incomparable by it.

## Where validation comes from

Not every clip is labelled in every camera, so the number of usable clips
depends on the view. Clips with labels, by view:

| Site-day | paired | rgb | thermal |
|---|---|---|---|
| MtErie | 0003, 0005, 0007 | 0003, 0005, 0007 | 0003, 0005, 0007 |
| Carnation | 0023 | 0023, 0025 | 0023 |
| FHL | 0134, 0401, 0405 | 0126, 0134, 0401, 0405, 0407 | 0134, 0401, 0403, 0405, 0564, 0566 |
| Baker | 1 | 1 | 1 |

So whenever it is a training site-day:

| Site-day | paired | rgb | thermal |
|---|---|---|---|
| MtErie | whole clip 0007 (172 frames) | whole clip 0007 | whole clip 0007 |
| Carnation | end of clip 0023, from frame 934 (111 frames), gap 250 | whole clip 0023 (739 frames) | end of clip 0023, from frame 934, gap 250 |
| FHL | whole clip 0134 (273 frames) | whole clip 0134 | whole clip 0134 |
| Baker | end of clip 1, from frame 1853 (327 frames), gap 250 | same | same |

The same rule holds in every fold, so a site-day's validation does not change
with the fold.

## The gap inside a clip

Frames are 0.2 s apart (5 per second; see `docs/wisard-pairing-review.md`),
and frame numbers count that time, holes included: Carnation 0023 has one
hole of 307 numbers. The gap is measured in frame numbers, so 250 is 50
seconds. Every training frame and every unlabelled frame of a clip is more
than 250 frame numbers from every validation frame of that clip; the check
command measures it from the image names (closest on the real manifests: 251).

Why 250 and not the 50 first proposed. `tools/frame_similarity_lags.py`
shrinks each frame to a 64x36 grey thumbnail and gives the mean correlation
of frames k positions apart (1 for identical frames), next to two references:
random pairs from the same clip, and random pairs from two site-days.

| Clip | Camera | 1 | 10 | 50 | 100 | 150 | 200 | 250 | Random pair, same clip |
|---|---|---|---|---|---|---|---|---|---|
| Baker 1 | rgb | 0.95 | 0.73 | 0.53 | 0.44 | 0.38 | 0.35 | 0.34 | 0.31 |
| Baker 1 | thermal | 0.95 | 0.71 | 0.49 | 0.41 | 0.36 | 0.34 | 0.31 | 0.31 |
| Carnation 0023 | rgb | 0.95 | 0.82 | 0.60 | 0.48 | 0.44 | 0.41 | 0.36 | 0.36 |
| Carnation 0023 | thermal | 0.99 | 0.91 | 0.73 | 0.60 | 0.57 | 0.54 | 0.47 | 0.47 |

Random pairs from Baker and Carnation score 0.12 (rgb) and 0.22 (thermal).
At 50 frames, frames of the two single-clip site-days are still far more
alike than two random frames of the clip. By 250 frames they are no more
alike than random frames of the same clip, which is the most a split inside
one site-day can achieve. The cost: Baker loses 250 of 2,180 frames, and
Carnation 0023 loses 250 of 739 in the paired and thermal views.

What the gap does not remove: the same people, terrain, and light are on both
sides. Validation inside a site-day is optimistic for that reason, and
should be used to choose settings, not quoted as a result. Only the test
fold is held out by site.

## Label fractions

An annotator labels a stretch of footage, not every hundredth frame. So each
training clip's labelled records, in frame order, are cut into blocks of 10.
Where the labels are unbroken a block is 2 seconds of footage, but a block
can span a hole in the labels or in the frame numbering, so it can cover
more time. The longest block, as the span from its first to its last frame
number, per fold:

| Fold (test site-day) | paired | rgb | thermal |
|---|---|---|---|
| MtErie | 9 (FHL 0405) | 148 (Carnation 0025) | 11 (FHL 0566) |
| Carnation | 28 (MtErie 0005) | 28 (MtErie 0005) | 28 (MtErie 0005) |
| FHL | 28 (MtErie 0005) | 148 (Carnation 0025) | 28 (MtErie 0005) |
| Baker | 28 (MtErie 0005) | 148 (Carnation 0025) | 28 (MtErie 0005) |

The 148 is the hole of 140 frame numbers in Carnation 0025, about 30
seconds; the 28 is the hole of 20 in MtErie 0005. Each site-day's blocks are shuffled with the seed; the site-days
are then interleaved in proportion to their number of blocks, so every
prefix holds them in proportion to within one block. A subset is the
shortest prefix holding at least its share of the frames, never less than
one block. Smaller subsets are prefixes of larger ones, so they are nested
by construction.

At 1% a fold has 3 to 8 blocks (30 to 80 labelled frames, about 6 to 16
seconds of footage where the blocks have no holes). There are not always enough blocks to give every training site-day
one: in five of the twelve fold-views the 1% subset holds two of the three
training site-days. At 5% every subset holds all three.

**Is 1% meaningful?** It is a real setting, a few seconds of footage per
site, and it is what a label shortage looks like. But with 3 to 8 blocks the
score will depend on which blocks the seed drew, so a 1% result needs
several seeds and their spread. 5% (130 to 370 frames, 13 to 37 blocks, every
training site-day present) is the smallest fraction where one seed is likely
to be representative.

## Is the same person at more than one site?

Unknown, and it cannot be settled from the labels. Crops of labelled people
from the RGB images show the same kinds of clothing recurring at MtErie,
Carnation and FHL (a white T-shirt with dark trousers, an orange or red top),
which suggests some of the same volunteers. Baker is in snow and everyone
wears winter clothing. So "an unseen site-day" means unseen terrain,
light, season and flight, but probably not always unseen people. A model could
gain from recognising a person's clothing from another site-day; nothing in
these folds prevents that.

## Leak routes checked

`aerial-search check-folds` reads the fold manifests and verifies the
following. Site-day, clip, and clip time are read again from each image's
directory and file name, using only the listed directories, never from the
fields the builder wrote:

- no image whose directory belongs to the test site-day appears in any
  training subset, validation set, or unlabelled pool of the fold, in any
  view; both cameras of a pair are checked, so the partner image of a test
  frame cannot slip in;
- the test set holds every labelled frame of its site-day in that view;
- training, validation, and test share no image within a view;
- each fraction is contained in the next, and 100% is the training set, so
  no subset is drawn from validation;
- the two images of a paired record are one moment of one clip;
- training and unlabelled frames are more than 250 frames from validation
  in the same clip. The 250 is written into the check, so a builder whose
  gap shrinks fails it whatever `folds.json` says;
- each label override is applied: its frames, found again in the source
  manifest, are in no manifest of their camera and are in the pool of every
  fold that trains on their site-day;
- the files are rebuilt byte for byte from the seed in `folds.json`.

Every fold writes files with the same names (`train_100pct.jsonl` and so on)
under its own directory. A cache must be keyed by the file's path or its
SHA-256, as `run.json` records it, never by its name alone.

A run on a fold records `fold` and `view` at the top level of its
`run.json`. A normal run refuses a checkpoint from a run on another fold or
view, or one that recorded none: pretraining in the MtErie fold has seen
Baker, so a detector scored on Baker must not start from it.

Validation of one view is not checked against training of another. In
Carnation 0023, the rgb view validates on the whole clip while the paired
view trains on its start. That is not a leak, because each view is a
separate experiment, but a run must take its training and validation from
the same view.

## What each fold holds

Generated by `aerial-search folds`. Frames are pairs in the paired view and
single images otherwise. Box size buckets are the edges in
`shared/aerial_search/src/aerial_search/evaluation/detection.py`, on the side of the box in
pixels of the image it is labelled on (RGB 3840x2160 at MtErie, 1920x1080
elsewhere; thermal 640x512).

### View: paired

| Fold (test site-day) | Split | Site-days | Clips | Frames | RGB boxes | Thermal boxes |
|---|---|---|---|---|---|---|
| MtErie | train | Carnation, FHL, Baker | 4 | 3,883 | 9,641 | 9,861 |
| MtErie | validation | Carnation, FHL, Baker | 3 | 711 | 1,296 | 925 |
| MtErie | test | MtErie | 3 | 708 | 1,770 | 1,824 |
| Carnation | train | MtErie, FHL, Baker | 5 | 3,978 | 9,495 | 10,013 |
| Carnation | validation | MtErie, FHL, Baker | 3 | 772 | 1,478 | 1,100 |
| Carnation | test | Carnation | 1 | 739 | 1,942 | 1,713 |
| FHL | train | MtErie, Carnation, Baker | 4 | 2,580 | 7,221 | 6,737 |
| FHL | validation | MtErie, Carnation, Baker | 3 | 610 | 1,442 | 1,006 |
| FHL | test | FHL | 3 | 2,112 | 4,044 | 4,867 |
| Baker | train | MtErie, Carnation, FHL | 5 | 2,816 | 6,994 | 7,568 |
| Baker | validation | MtErie, Carnation, FHL | 3 | 556 | 554 | 620 |
| Baker | test | Baker | 1 | 2,180 | 6,012 | 5,261 |

Validation source per training site-day:

- fold MtErie: Carnation: clip 0023 from frame 934, 250 frames dropped before it; FHL: whole clip 0134; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold Carnation: MtErie: whole clip 0007; FHL: whole clip 0134; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold FHL: MtErie: whole clip 0007; Carnation: clip 0023 from frame 934, 250 frames dropped before it; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold Baker: MtErie: whole clip 0007; Carnation: clip 0023 from frame 934, 250 frames dropped before it; FHL: whole clip 0134

Test boxes by size (side of the box in pixels, very_tiny [0, 8), tiny [8, 16), small [16, 32), medium_plus [32, inf)):

| Fold | Camera | very_tiny | tiny | small | medium_plus |
|---|---|---|---|---|---|
| MtErie | rgb | 0 | 0 | 0 | 1,770 |
| MtErie | thermal | 0 | 419 | 915 | 490 |
| Carnation | rgb | 0 | 0 | 774 | 1,168 |
| Carnation | thermal | 613 | 1,002 | 97 | 1 |
| FHL | rgb | 0 | 164 | 2,118 | 1,762 |
| FHL | thermal | 193 | 2,645 | 1,801 | 228 |
| Baker | rgb | 0 | 0 | 27 | 5,985 |
| Baker | thermal | 0 | 95 | 1,263 | 3,903 |

Label fractions, frames / rgb boxes + thermal boxes / site-days:

| Fold | 1% | 5% | 10% | 100% |
|---|---|---|---|---|
| MtErie | 40 / 84 + 103 / 2 | 200 / 438 + 461 / 3 | 390 / 974 + 1,001 / 3 | 3,883 / 9,641 + 9,861 / 3 |
| Carnation | 40 / 105 + 112 / 3 | 200 / 486 + 465 / 3 | 398 / 985 + 976 / 3 | 3,978 / 9,495 + 10,013 / 3 |
| FHL | 30 / 55 + 44 / 2 | 130 / 298 + 256 / 3 | 260 / 689 + 612 / 3 | 2,580 / 7,221 + 6,737 / 3 |
| Baker | 30 / 88 + 127 / 2 | 141 / 408 + 437 / 3 | 291 / 918 + 926 / 3 | 2,816 / 6,994 + 7,568 / 3 |

### View: rgb

| Fold (test site-day) | Split | Site-days | Clips | Frames | RGB boxes |
|---|---|---|---|---|---|
| MtErie | train | Carnation, FHL, Baker | 6 | 6,799 | 16,688 |
| MtErie | validation | Carnation, FHL, Baker | 3 | 1,339 | 3,126 |
| MtErie | test | MtErie | 3 | 708 | 1,770 |
| Carnation | train | MtErie, FHL, Baker | 7 | 6,020 | 12,917 |
| Carnation | validation | MtErie, FHL, Baker | 3 | 772 | 1,478 |
| Carnation | test | Carnation | 2 | 2,054 | 7,189 |
| FHL | train | MtErie, Carnation, Baker | 4 | 3,454 | 10,846 |
| FHL | validation | MtErie, Carnation, Baker | 3 | 1,238 | 3,272 |
| FHL | test | FHL | 5 | 4,154 | 7,466 |
| Baker | train | MtErie, Carnation, FHL | 7 | 5,732 | 14,041 |
| Baker | validation | MtErie, Carnation, FHL | 3 | 1,184 | 2,384 |
| Baker | test | Baker | 1 | 2,180 | 6,012 |

Validation source per training site-day:

- fold MtErie: Carnation: whole clip 0023; FHL: whole clip 0134; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold Carnation: MtErie: whole clip 0007; FHL: whole clip 0134; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold FHL: MtErie: whole clip 0007; Carnation: whole clip 0023; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold Baker: MtErie: whole clip 0007; Carnation: whole clip 0023; FHL: whole clip 0134

Test boxes by size (side of the box in pixels, very_tiny [0, 8), tiny [8, 16), small [16, 32), medium_plus [32, inf)):

| Fold | Camera | very_tiny | tiny | small | medium_plus |
|---|---|---|---|---|---|
| MtErie | rgb | 0 | 0 | 0 | 1,770 |
| Carnation | rgb | 2 | 888 | 2,900 | 3,399 |
| FHL | rgb | 2 | 229 | 2,744 | 4,491 |
| Baker | rgb | 0 | 0 | 27 | 5,985 |

Label fractions, frames / rgb boxes / site-days:

| Fold | 1% | 5% | 10% | 100% |
|---|---|---|---|---|
| MtErie | 70 / 117 / 3 | 345 / 939 / 3 | 685 / 1,746 / 3 | 6,799 / 16,688 / 3 |
| Carnation | 70 / 154 / 3 | 310 / 546 / 3 | 607 / 1,171 / 3 | 6,020 / 12,917 / 3 |
| FHL | 40 / 78 / 3 | 180 / 548 / 3 | 350 / 1,088 / 3 | 3,454 / 10,846 / 3 |
| Baker | 60 / 228 / 3 | 290 / 727 / 3 | 580 / 1,499 / 3 | 5,732 / 14,041 / 3 |

### View: thermal

| Fold (test site-day) | Split | Site-days | Clips | Frames | Thermal boxes |
|---|---|---|---|---|---|
| MtErie | train | Carnation, FHL, Baker | 7 | 5,990 | 13,073 |
| MtErie | validation | Carnation, FHL, Baker | 3 | 712 | 926 |
| MtErie | test | MtErie | 3 | 708 | 1,824 |
| Carnation | train | MtErie, FHL, Baker | 8 | 6,085 | 13,225 |
| Carnation | validation | MtErie, FHL, Baker | 3 | 772 | 1,100 |
| Carnation | test | Carnation | 1 | 740 | 1,714 |
| FHL | train | MtErie, Carnation, Baker | 4 | 2,580 | 6,737 |
| FHL | validation | MtErie, Carnation, Baker | 3 | 611 | 1,007 |
| FHL | test | FHL | 6 | 4,219 | 8,079 |
| Baker | train | MtErie, Carnation, FHL | 8 | 4,923 | 10,780 |
| Baker | validation | MtErie, Carnation, FHL | 3 | 557 | 621 |
| Baker | test | Baker | 1 | 2,180 | 5,261 |

Validation source per training site-day:

- fold MtErie: Carnation: clip 0023 from frame 934, 250 frames dropped before it; FHL: whole clip 0134; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold Carnation: MtErie: whole clip 0007; FHL: whole clip 0134; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold FHL: MtErie: whole clip 0007; Carnation: clip 0023 from frame 934, 250 frames dropped before it; Baker: clip 1 from frame 1853, 250 frames dropped before it
- fold Baker: MtErie: whole clip 0007; Carnation: clip 0023 from frame 934, 250 frames dropped before it; FHL: whole clip 0134

Test boxes by size (side of the box in pixels, very_tiny [0, 8), tiny [8, 16), small [16, 32), medium_plus [32, inf)):

| Fold | Camera | very_tiny | tiny | small | medium_plus |
|---|---|---|---|---|---|
| MtErie | thermal | 0 | 419 | 915 | 490 |
| Carnation | thermal | 613 | 1,002 | 98 | 1 |
| FHL | thermal | 235 | 3,111 | 3,762 | 971 |
| Baker | thermal | 0 | 95 | 1,263 | 3,903 |

Label fractions, frames / thermal boxes / site-days:

| Fold | 1% | 5% | 10% | 100% |
|---|---|---|---|---|
| MtErie | 60 / 119 / 3 | 300 / 600 / 3 | 603 / 1,381 / 3 | 5,990 / 13,073 / 3 |
| Carnation | 70 / 165 / 3 | 306 / 689 / 3 | 616 / 1,372 / 3 | 6,085 / 13,225 / 3 |
| FHL | 30 / 58 / 2 | 130 / 328 / 3 | 260 / 685 / 3 | 2,580 / 6,737 / 3 |
| Baker | 50 / 82 / 2 | 250 / 544 / 3 | 501 / 1,103 / 3 | 4,923 / 10,780 / 3 |

### Unlabelled pool

| Fold | Site-days | Clips | Pairs |
|---|---|---|---|
| MtErie | Carnation, Hannegan, FHL, Baker | 12 | 12,537 |
| Carnation | MtErie, Hannegan, FHL, Baker | 13 | 11,760 |
| FHL | MtErie, Carnation, Hannegan, Baker | 6 | 4,588 |
| Baker | MtErie, Carnation, Hannegan, FHL | 13 | 11,470 |

## What four site-days can and cannot support

Four folds give four test scores per method, each from one site on one day.
They can show whether a method beats its baseline at every site, or only at
some, and roughly how much scores move from one site to the next. They cannot
give a confidence interval worth the name: the spread of four numbers is a
description, not an estimate, and the sites differ in more ways than one
(Baker is snow with large people; Carnation's thermal people are almost all
under 16 pixels; MtErie has no small RGB people at all). A size bucket only
says something on folds whose test set holds people of that size, and the
table above shows which those are. So a claim takes the form "better at all
four sites, by these margins", or "better at three of four", never "better
on average at an unseen site-day" with an error bar. More site-days, from
another dataset, are the only way to a stronger claim.

## Regenerate and check

```bash
uv run aerial-search prepare data/raw/wisard-full --output data/manifests/wisard-full
uv run aerial-search folds data/raw/wisard-full data/manifests/wisard-full
uv run aerial-search check-folds data/raw/wisard-full data/manifests/wisard-full
```

`folds` replaces an existing `folds/` that it wrote before (it holds
`folds.json`) and refuses any other non-empty directory. It reads the raw
images only for their sizes. Each command takes about ten seconds.
