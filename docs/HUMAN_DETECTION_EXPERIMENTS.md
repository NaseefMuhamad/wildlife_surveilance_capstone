# Human detection experiments

The epoch 9 checkpoint from `runs/birdsai_corrected_20261114_234448_294790`
was evaluated on the corrected validation split. TestReal was reserved from
these experiments. The existing TestReal comparison remains the baseline.

## Confidence, resolution and missed boxes

`scripts/probe_human_detection.py` sampled every eighth annotated validation
frame within each of the five sequences. The 390 sampled frames contained
334 human annotations. Human predictions were matched one-to-one at IoU 0.50.
The script saves the complete confidence sweep and breakdown by sequence and
box height in `outputs/evaluation/human_validation_probe.json`.

| Input size | Confidence | Correct human boxes | False human boxes |
|---|---:|---:|---:|
| 640 | 25% | 0 | 35 |
| 640 | 5% | 7 | 203 |
| 640 | 1% | 12 | 541 |
| 960 | 25% | 0 | 4 |
| 960 | 5% | 6 | 30 |
| 960 | 1% | 11 | 90 |

At 640 and 25% confidence, 31 human annotations matched an animal prediction.
Of the remaining annotations, 12 matched a lower-confidence human box, four
had partial overlap with a human box at IoU between 0.10 and 0.50, and 287
fell outside these categories. One sequence contained all 324 annotations
with box height below 16 pixels at the 640 input scale. A second sequence
contained all ten larger annotations. Size and sequence effects are
confounded in this sample.

## Training annotation audit

The training split contains 9,834 human boxes from seven sequence-local human
track IDs across four of 27 sequences. The two largest human sequences supply
8,126 boxes (82.6%). The validation split contains seven sequence-local human
track IDs across two sequences. All raw rows with species ID 0 map to class 1
in the audited TrainReal split, and all class 1 rows have species ID 0.
This confirms agreement between the two annotation fields. A visual review
would be needed to evaluate annotation accuracy.

## Track confirmation probe

For each of the two human validation sequences, the probe replayed up to 24
annotated frames starting at each of four evenly spaced positions in the list
of human frame IDs. It reset the existing `DetectionTracker` between windows
and across gaps in the exported frame IDs. At 640 input size, an alert fired
once per predicted track after one or two detections. Correct alerts matched
a ground-truth human box at the trigger frame at IoU 0.50.

| Confidence | Detections required | Human track IDs caught | Alerts | Unmatched at trigger |
|---|---:|---:|---:|---:|
| 25% | 1 | 1 of 7 | 17 | 15 |
| 25% | 2 | 0 of 7 | 9 | 9 |
| 10% | 1 | 2 of 7 | 39 | 33 |
| 10% | 2 | 1 of 7 | 24 | 22 |

The probe measures trigger-frame matches in short windows. Continuous camera
recordings with verified empty intervals are needed to estimate false alerts
per camera-hour.

## Next controlled experiment

Collect labeled footage from the intended thermal camera across separate
recordings, including people at different distances and verified empty
scenes. Hold out entire recordings for validation and final testing. Compare
the frozen checkpoint with a fine-tuned checkpoint using human-rich training
batches and a matched fine-tuning control. Measure human track coverage, time
to first correct alert, and false alerts per camera-hour. Training with
additional camera data and the controlled sampling comparison remain pending.

## Small-object tile pilot

`scripts/human_tiles.py --prepare` builds an alternate version of the
1,500-frame real-only pilot. It uses 750 full frames and 750 crops, of which
252 contain people. A crop covers 60% of the original width and height.
The four candidate positions overlap by 20% of the original dimensions.
For a selected frame with people, preparation chooses the crop that retains
at least 80% of the area of the most human annotations. It clips all
intersecting annotations to the crop boundaries. The resulting dataset has
798 human boxes and 2,570 animal boxes, and retains the original real
validation frames. Increasing human scale also changes the visible context
and the number of animal examples.

Evaluation samples every fourth frame in each of the five validation
sequences and compares full-frame inference, four-tile inference, and the two
combined. It maps tile boxes into the original image and suppresses duplicate
human predictions at IoU 0.50. Human predictions are matched one-to-one to
human labels at IoU 0.50. The confidence sweep covers 5% through 60%. The
sample has 776 frames with 672 human instances, including 652 with height
below 16 pixels at the 640-pixel input scale.

| Checkpoint and inference at 25% confidence | Correct humans | Correct small humans | False humans |
|---|---:|---:|---:|
| Epoch 9, full frame | 1 | 1 | 73 |
| Epoch 9, tiles | 3 | 3 | 21 |
| Epoch 9, full frame and tiles | 4 | 4 | 90 |
| Real-only epoch 3, tiles | 1 | 1 | 8 |
| Crop-trained epoch 3, tiles | 0 | 0 | 183 |
| Simulation best, full frame | 16 | 9 | 496 |
| Simulation best, tiles | 7 | 5 | 193 |

The crop-trained run is `runs/birdsai_human_tiles_20261117_140515_794164`.
It starts from epoch 9 and uses three epochs, batch 2, and image size 640 on
Intel Arc. In-training validation was deferred to the last epoch and its
NMS ran on CPU. The sampled comparison uses `last.pt` for both real-only
and crop-trained pilots and the previously selected `best.pt` for the
simulation pilot. All inference comparisons use the same evaluation code.

On all 3,098 real validation frames, original epoch 9 with full-frame
inference at 25% confidence matched 6 humans, 5 small, with 324 false
human boxes. Four-tile inference at 10% confidence matched 20 humans,
19 small, with 193 false human boxes. At the same 25% confidence, tiles
matched 8 humans, all small, with 85 false human boxes. These counts use
human-only prediction and matching, so the earlier Ultralytics confusion
matrix counts differ. The reference for this comparison is the full-frame
result produced by the tile evaluator on the same frames.

The interrupted TestReal evaluation was rerun to completion on November 17.
Every fourth frame per sequence yielded 3,879 frames with 5,469 human boxes.
The fixed validation choice of tiles at 10% confidence matched 226 humans
with 1,345 false human boxes. Full frames at 25% confidence matched 298
humans with 502 false human boxes in the same evaluator. Tiling's
validation improvement did not transfer to this test sample.
Evidence is saved in
`outputs/evaluation/human_tiles_20261117_210804_138692.json`.

## Explicit low learning rate pilot

`scripts/low_lr_pilot.py` starts from epoch 9 on the same 1,500 real-only
training frames. It sets AdamW explicitly with initial learning rate
0.00005, final rate fraction 0.1, and zero warmup, for three epochs.
Earlier short pilots used automatic optimizer selection and their first
epochs logged learning rates around 0.0008 to 0.0011. This experiment
changes both learning rate and warmup. It tests whether gentler updates
avoid the human detection degradation observed in the earlier pilots.

The completed run is `runs/birdsai_low_lr_20261117_210816_059463`.
Fixed-threshold evaluation of its `last.pt` on all 3,098 real validation
frames matched 7 of 2,697 human boxes, with 290 false human detections.
It matched 2 of 2,618 small human boxes. At the same settings, original epoch
9 matched 11 human boxes, all small, with 260 false detections. Lowering the
learning rate and removing warmup did not outperform the starting checkpoint
in this three-epoch pilot. Results are in
`outputs/evaluation/human_pilot_fixed_20261117_213153_070678/fixed_results.json`.

## Freezing feature parameters

`scripts/low_lr_pilot.py --freeze 10` uses the same low-rate settings but
freezes the first ten YOLOv8 modules. The run
`runs/birdsai_low_lr_freeze10_20261117_213728_031938` matched 13 of 2,697
human boxes at 25% confidence and IoU 0.50, with 418 false detections.
All 13 matches were among 2,618 small human boxes. This improves recall
slightly relative to epoch 9 (11 matches), with more false detections
(epoch 9 had 260). The evidence is saved in
`outputs/evaluation/human_pilot_fixed_20261117_214702_552610/fixed_results.json`.

A visual spot-check of 12 real labeled crops showed several validation
human targets with native heights of 3 to 12 pixels. The local inspection
image is `outputs/evaluation/human_annotation_audit.jpg`. The inspection
does not establish annotation accuracy across the complete dataset.

The stronger `--freeze 22` pilot adapts the detection head while freezing
the first 22 modules' parameters. Run
`runs/birdsai_low_lr_freeze22_20261117_215119_581972` matched 15 humans,
all small, with 445 false human detections. Its recall was 0.556% and
precision 3.26% at 25% confidence and IoU 0.50. It adds four correct
human boxes and 185 false detections relative to original epoch 9.
This single-seed result gives no basis to claim a robust improvement.
Evidence is saved in
`outputs/evaluation/human_pilot_fixed_20261117_215947_858396/fixed_results.json`.

Both freeze pilots use the existing real-only 1,500-frame subset, three
epochs, explicit AdamW at 0.00005, and no warmup. They preserve the
training source sequences and real validation footage. Freezing refers
to trainable parameters using the installed Ultralytics implementation.
This experiment does not assert identical normalization buffers.

## Memorization sanity check

`scripts/human_sanity_check.py` selects four human-containing frames with the
largest minimum human box height from each of the four real training
sequences. The 16 frames contain 19 human boxes. Evaluation intentionally
uses these same frames to check whether the model can fit known examples.
It is excluded from generalization comparisons.

Epoch 9 matched 13 of 19 human boxes, with five false human detections, at
confidence 0.25 and matching IoU 0.50. After 100 epochs on just these 16
frames, the diagnostic checkpoint matched 19 of 19, with zero false human
detections at the same thresholds. Training used Intel Arc, batch 2,
AdamW at 0.001, zero warmup, and disabled mosaic, geometric and color
augmentation. It completed in approximately 4.3 minutes of trainer time.

The result demonstrates that this model and training path can learn the
selected real human annotations. It does not validate all annotations,
small-human transfer, animal retention, or performance on unseen footage.
The diagnostic weights are retained under
`outputs/diagnostics/human_memorization_20261117_220433_220079/training/weights`.
The exact before/after counts are in the adjacent `sanity_results.json`.
The labeled-frame preview was visually inspected for box alignment.
