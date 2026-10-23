# WiSARD RGB-thermal pairing review

Issue #2. How RGB (VIS) frames are paired with thermal (IR) frames, the
evidence that the pairing is right, and what was left out.

## Result

Regenerate the manifests with:

```bash
uv run aerial-search prepare data/raw/wisard-full --output data/manifests/wisard-full
```

On the raw data in the main checkout this gives **14,834 pairs, 5,739 of them
labelled in both cameras**, from 17 VIS/IR clip pairs recorded on five flight
days with a DJI Mavic 2 Enterprise Advanced. `data_quality.json` next to the
manifests lists, per clip, the pairs by which camera is labelled, the boxes
clipped to the image, and every image directory that was not paired. Numbers
quoted elsewhere must come from a run on `main`.

| File | Contents |
|---|---|
| `all_pairs.jsonl` | every pair, labelled or not (for self-supervised learning) |
| `full.jsonl` | pairs labelled in both cameras |
| `rgb_labelled.jsonl`, `thermal_labelled.jsonl` | every frame of a listed clip with a label file for that camera, whether or not its partner has one (for detection on one camera) |

A listed directory that is missing stops the run with an error. To prepare a
subset on purpose, name the clips:
`uv run aerial-search prepare <root> --output <dir> --collection <id> ...`.

## The pairing rule

1. **Clips.** Only the VIS/IR directory pairs listed in `WISARD_COLLECTIONS`
   (`shared/aerial_search/src/aerial_search/data/wisard.py`) are paired. The Mavic 2 Enterprise
   writes the VIS and IR video of one recording as consecutive DJI clip
   numbers, so `..._VIS_0003` goes with `..._IR_0004`.
   `220109_Baker_Enterprise_VIS_1` holds `DJI_0582` and `IR_1` holds
   `DJI_0583`. Each entry records both clip numbers, and every image name in a
   directory must carry its clip number, or the run stops. This holds for all
   34 listed directories, and it rejects any pairing of two different clips.
   Anything not listed is reported as unpaired, never guessed.
2. **Frames.** RGB frame i pairs with thermal frame i + `thermal_offset`. The
   offset is 0 for every clip except `MtErie_0003`, where it is 1: its VIS
   frames are numbered 0 to 263 and its IR frames 1 to 264. The offset comes
   from where the numbering starts, not from a fitted lag. The frame number is
   the last run of digits after a `_` or space before `.jpg`/`.jpeg`, of any
   length. A name with no frame number, or two images with one frame number
   in a directory, raises an error.
3. **Coverage.** At least 95% of the frames of each directory must find a
   partner, or the run stops. In the data the lowest share is 907 of 909. This
   catches numbering that does not line up, such as a missing offset or clips
   at different frame rates (Airfield VIS_4 has 572 frames, IR_4 has 1,061).
   It does not catch two different clips of similar length numbered from 0.
   Of the 115 wrong VIS/IR pairings on the same flight days, 11 pass it (106
   passed the first version, which measured against the smaller directory).
   The rule that IR clip = VIS clip + 1 for every listed entry, which a test
   enforces, rejects all 115. The file-name check in rule 1 only proves that
   files belong to their directory.

Every naming scheme in the dataset, and what the number is:

| Scheme | Example | Used in pairs |
|---|---|---|
| `<dir>_<8 digits>` | `210417_MtErie_Enterprise_VIS_0003_00000263.jpeg` | yes |
| `DJI_<n>_<8 digits>` | `DJI_0053_00000715.jpg` | yes |
| `<name>.mp4_<5 digits>` | `DJI_0402.mp4_00000.jpg` | yes |
| `<dir>_<5 digits>`, `<dir>_out_frame_<5 digits>` | `..._IR_0008_out_frame_00001.jpeg` | yes |
| `<dir>_<1-4 digits>` | `200402_Karen_Inspire_VIS_695.jpeg` | no, VIS only |
| `<date>_<time>_IR <3 digits>` | `20200929_134258_IR 127.jpg` | no, IR only |
| `<name>_ <4 digits>` | `200704_baker__FLIR_1_ 0001.jpg` | no, IR only |

In all of them the number is a frame counter within one exported clip. The
`IR 127` scheme counts up to at most 452 per directory, and the time in the
name is the clip's start time. Those directories have no VIS partner, so the
question of what the number means does not affect any pair.

## Evidence

**The paper** (Broyles et al., IROS 2022, arXiv 2309.04453, Section III-B):
video was recorded at 30 Hz and every sixth frame was kept, so neighbouring
frames are 0.2 s apart. The Mavic 2 Enterprise visual stream runs at about
29.95 Hz, "resulting in a one or two image difference between corresponding
multi-modal image sets". Pairs are defined as "taken at the same instant in
time". The frame numbers count kept frames, not 30 Hz frames: the seven
`210327_Airfield_FLIR_IR_*` clips have their start times in their names; the
clips starting 11:47:12, 11:50:44, 11:57:41 and 12:01:13 each hold 1,059 to
1,062 frames and are followed by the next clip about 212 s later, which is 5
frames per second. No pairing file ships with the dataset; `WiSARDv1.zip`
holds only the image directories.

**Camera motion.** A pan moves the whole image in both cameras at once. The
shift between consecutive frames, by phase correlation, is a signal both
directories share. `tools/pairing_offsets.py lags` finds the lag at which the
VIS and IR signals correlate best (+1: VIS frame i matches IR frame i+1). Lags
are measured from equal frame numbers, before the `MtErie_0003` offset:

| Collection | Whole | First third | Middle third | Last third |
|---|---|---|---|---|
| 210417_MtErie_Enterprise_0003 | +1 (r=0.66) | +1 (r=0.64) | +1 (r=0.91) | +1 (r=0.68) |
| 210417_MtErie_Enterprise_0005 | +1 (r=0.64) | +0 (r=0.65) | +0 (r=0.64) | +1 (r=0.54) |
| 210417_MtErie_Enterprise_0007 | +0 (r=0.80) | +0 (r=0.88) | +0 (r=0.72) | +0 (r=0.86) |
| 210529_Carnation_Enterprise_0023 | +1 (r=0.66) | +0 (r=0.90) | +0 (r=0.75) | +1 (r=0.79) |
| 210529_Carnation_Enterprise_0025 | +2 (r=0.35) | +1 (r=0.56) | +1 (r=0.62) | +2 (r=0.38) |
| 210812_Hannegan_Enterprise_0053 | +0 (r=0.54) | +0 (r=0.71) | +0 (r=0.64) | +0 (r=0.52) |
| 210812_Hannegan_Enterprise_0055 | +1 (r=0.81) | +0 (r=0.78) | +0 (r=0.49) | +1 (r=0.82) |
| 210924_FHL_Enterprise_0126 | +1 (r=0.85) | +1 (r=0.92) | +1 (r=0.91) | +1 (r=0.75) |
| 210924_FHL_Enterprise_0134 | -1 (r=0.41) | -1 (r=0.29) | +0 (r=0.97) | -10 (r=0.55) |
| 210924_FHL_Enterprise_0401 | +1 (r=0.65) | +0 (r=0.86) | +1 (r=0.69) | +1 (r=0.56) |
| 210924_FHL_Enterprise_0403 | +1 (r=0.77) | +0 (r=0.87) | +1 (r=0.77) | +2 (r=0.86) |
| 210924_FHL_Enterprise_0405 | +0 (r=0.55) | +0 (r=0.48) | +0 (r=0.84) | -1 (r=0.40) |
| 210924_FHL_Enterprise_0407 | +1 (r=0.62) | +1 (r=0.71) | +1 (r=0.58) | +2 (r=0.62) |
| 210924_FHL_Enterprise_0409 | +1 (r=0.81) | +0 (r=0.72) | +1 (r=0.94) | +1 (r=0.79) |
| 210924_FHL_Enterprise_0564 | +0 (r=0.57) | +0 (r=0.60) | +1 (r=0.60) | +1 (r=0.70) |
| 210924_FHL_Enterprise_0566 | +1 (r=0.11) | +0 (r=0.91) | +1 (r=0.17) | +2 (r=0.08) |
| 220109_Baker_Enterprise_1 | +0 (r=0.77) | -1 (r=0.79) | -1 (r=0.93) | +0 (r=0.84) |

**What the tool can and cannot show.** It measures shifts in whole pixels on
frames shrunk to 160 pixels wide, so most frame-to-frame shifts read as zero
and the correlation rests on a few bursts of fast motion. It separates right
clips from wrong ones: as a control, all 82 wrong VIS/IR directory pairs from
the same flight days score at most r=0.16 within two frames of lag 0. It
cannot choose between neighbouring lags: for Baker, which holds 38% of
labelled pairs, r is 0.74 at lag -1 and 0.77 at lag 0. The control shows only
the first of these. So the table supports "the right clips, within about two
frames" and nothing finer. The lags in it agree with the paper's 29.95 Hz
note but do not measure the drift. The one-frame offset of `MtErie_0003`
rests on its numbering, which starts one apart; the reviewer of PR #36
measured the same offset independently from the motion of labelled box
centres.

Weak rows, read with care:
- `FHL_0566` is at dusk. The VIS frames go black after the first few
  hundred, so there is no VIS signal to correlate. The start is aligned
  (r=0.91 at lag 0).
- `FHL_0134` last third (-10) is a hover with almost no motion, and the
  contact sheet shows the same two people on the same trail at the end.

**Contact sheets**, one per collection, in `outputs/pairing-check/` in the
main checkout (`tools/pairing_contact_sheets.py`). Each shows two pairs each
from the start, middle, and end, with boxes. In every one of the 17 the RGB
and thermal images show the same scene at all three points: the same people
in the same arrangement where there are labels, and the same trails,
shorelines, and tree lines where there are not. The thermal camera has a
narrower view, so people sit nearer the edge of the thermal frame. A lag of
one or two frames (0.2 to 0.4 s) is not visible by eye on these sheets. The
`MtErie_0003` sheet was redrawn after the offset: VIS frame 0 now pairs with
IR frame 1, and it shows the same scene and people at start, middle, and end.

## Labels

Pairs by which camera has a label file, and boxes clipped to the image, per
clip (from `data_quality.json`):

| Clip | Pairs | Both | RGB only | Thermal only | Neither | Boxes clipped RGB / IR |
|---|---|---|---|---|---|---|
| 210417_MtErie_Enterprise_0003 | 264 | 264 | 0 | 0 | 0 | 0 / 0 |
| 210417_MtErie_Enterprise_0005 | 272 | 272 | 0 | 0 | 0 | 6 / 15 |
| 210417_MtErie_Enterprise_0007 | 172 | 172 | 0 | 0 | 0 | 1 / 0 |
| 210529_Carnation_Enterprise_0023 | 739 | 739 | 0 | 0 | 0 | 5 / 7 |
| 210529_Carnation_Enterprise_0025 | 1,313 | 0 | 1,313 | 0 | 0 | 33 / 0 |
| 210812_Hannegan_Enterprise_0053 | 714 | 0 | 0 | 0 | 714 | 0 / 0 |
| 210812_Hannegan_Enterprise_0055 | 422 | 0 | 0 | 0 | 422 | 0 / 0 |
| 210924_FHL_Enterprise_0126 | 1,006 | 0 | 1,004 | 0 | 2 | 0 / 0 |
| 210924_FHL_Enterprise_0134 | 273 | 273 | 0 | 0 | 0 | 0 / 0 |
| 210924_FHL_Enterprise_0401 | 896 | 896 | 0 | 0 | 0 | 19 / 17 |
| 210924_FHL_Enterprise_0403 | 1,417 | 0 | 0 | 1,417 | 0 | 0 / 10 |
| 210924_FHL_Enterprise_0405 | 943 | 943 | 0 | 0 | 0 | 0 / 16 |
| 210924_FHL_Enterprise_0407 | 1,035 | 0 | 1,035 | 0 | 0 | 11 / 0 |
| 210924_FHL_Enterprise_0409 | 1,299 | 0 | 0 | 0 | 1,299 | 0 / 0 |
| 210924_FHL_Enterprise_0564 | 982 | 0 | 0 | 982 | 0 | 0 / 16 |
| 210924_FHL_Enterprise_0566 | 907 | 0 | 0 | 907 | 0 | 0 / 1 |
| 220109_Baker_Enterprise_1 | 2,180 | 2,180 | 0 | 0 | 0 | 116 / 278 |
| **Total** | **14,834** | **5,739** | **3,352** | **3,306** | **2,437** | **191 / 360** |

6,658 pairs are labelled in one camera only, more than the 5,739 labelled in
both. `full.jsonl` keeps only the pairs labelled in both. The per-camera
manifests carry every labelled frame: 9,096 RGB and 9,048 thermal. Apart from
two frames of `FHL_0126`, a clip is labelled in a camera throughout or not at
all.

Boxes are clipped to the image. 551 boxes ran past an edge and were clipped,
278 of them in `Baker_IR_1`. None lost all its area, so none was dropped.
Labels are written to six decimals, so a box drawn to the edge can overshoot
by up to 1e-6; 22 such boxes are clipped but not counted. Each label file is
read once, so no box is counted twice.

## Excluded

| Directories | Frames (VIS / IR) | Reason |
|---|---|---|
| `210327_Airfield_FLIR_VIS_1`-`4`, `IR_1`-`8` | 1,620 / 6,874 | VIS and IR frame numbers run at different rates. For VIS_3/IR_3 and VIS_4/IR_4 the motion signals match only when IR frame = 1.834 x VIS frame + 6 to 9 (r=0.59, 0.75, `tools/pairing_offsets.py scale`). VIS_1 and VIS_2 match no IR directory at any rate tried (best r=0.26 to 0.31). Equal frame numbers are the same moment only near the start; the sheets in `outputs/pairing-check/excluded/` show different scenes in the middle and at the end |
| `220109_Baker_Enterprise_IR_2` | 0 / 609 | `DJI_0585`; its VIS file `DJI_0584` is not in the dataset |
| 33 other directories | 9,788 / 7,553 | One camera only: Phantom, Inspire and Mavic Mini flights are VIS only; `200704_Baker`, `200910_Carnation` and `200929_Karen` FLIR are IR only |

The Airfield pairs could be recovered with a rate-scaled mapping, but the
mapping is fitted, not documented, so they are left out.

## The two old manifest sets

Both totals were reproduced exactly by running the two old versions of the
code on the same raw data.

| Set | Code | All | Labelled | What went wrong |
|---|---|---|---|---|
| `data/manifests/processed/wisard-full` (deleted) | before `a31faf5`: pairs by position in sorted order | 16,459 | 7,359 | Includes Airfield, 1,620 pairs, all labelled, paired the k-th VIS frame with the k-th IR frame. Position pairing also slips where one directory skips a frame the other has: in `Carnation_0025` pairs after VIS frame 654 are up to 140 frame numbers apart, in `FHL_0566` up to 3 |
| `data/manifests/wisard-full` (replaced) | `main` before this fix: frame numbers of 5 or 6 digits only | 8,759 | 4,021 | 8-digit numbers were not parsed, so 7 directory pairs collapsed to one arbitrary pair each, and `MtErie_0005`, `MtErie_0007`, `FHL_0134` produced none |

The new set is the old positional set minus Airfield (1,620 pairs, 1,620
labelled) and minus 5 frames that have no partner. `MtErie_0003` is paired as
the positional set paired it, VIS frame i with IR frame i+1.

The paper counts 15,453 pairs. This set has 14,834, and 16,454 with Airfield.
The paper's own tables do not add up to that figure (the terrain table's
multi-modal column sums to 27,094), so it is a loose reference.

## Lead's decisions, 2026-09-29

- Pairs 0 to 2 frames (0 to 0.4 s) apart are accepted as they are, for
  self-supervised learning and for detection on one camera at a time. No lag
  is fitted per clip; the offset tool cannot support one.
- `MtErie_0003` is corrected by its numbering (`thermal_offset=1`).
- `FHL_0134` and `FHL_0566` stay in.
- Airfield stays out.
- Any later stage that moves boxes from one camera to the other must drop or
  down-weight frames in fast motion (epic #16). For scale: a walking person
  moves about one box width in 0.4 s, and during a pan two frames are 50 to
  110 pixels of image motion.

## Open

- **Splits.** With these pairs, seed 7 puts no labelled pair in `test.jsonl`,
  because `220109_Baker_Enterprise_1` holds 38% of labelled pairs and the
  greedy split fills train and validation first. Collections are single clips,
  so clips from one site and day can land in different splits. Split logic
  was out of scope. Replaced by site-day folds in #3; see
  `docs/site-folds-review.md`.
