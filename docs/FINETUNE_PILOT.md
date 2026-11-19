# Reserve Watch fine-tuning pilot

Both arms start from the same epoch 9 checkpoint and use the same 1,500-frame
training budget, identical real validation data, and three epochs on Intel Arc.
This is a subset experiment. Each arm shares 500 real frames with human labels and
665 real animal-only frames, selected across sequences and spread through each
recording. The control adds 335 more real animal-only frames. The second arm
uses the 335 human-positive BIRDSAI simulation frames in those slots. The
experiment measures the effect of that substitution at a fixed training budget.
It changes both the source of those frames and the class mix.

The prepared datasets are under `data/finetune_pilot_v1/`. Each arm has a
manifest with the selected frames and validation label hash. The preparation
script checks for image bytes shared between pilot training and either real
validation or TestReal. TestReal stays in the original dataset and is excluded
from both pilot configurations. Treat the exported images as read-only because
they can be hardlinked to the original files.

## Run in Command Prompt

Run from the backend repository root. The existing training launcher selects
the configured Intel Arc interpreter from `.xpu-python`. Run the commands one
at a time. The first epoch is the time estimate for the rest of the pilot.

```cmd
set "START=runs/birdsai_corrected_20261114_234448_294790/weights/best.pt"
.venv\Scripts\python.exe src\train.py --data data/finetune_pilot_v1/real_only/dataset.yaml --model "%START%" --device xpu:0 --epochs 3 --patience 0 --batch 2 --imgsz 640 --workers 0
.venv\Scripts\python.exe src\train.py --data data/finetune_pilot_v1/real_plus_sim/dataset.yaml --model "%START%" --device xpu:0 --epochs 3 --patience 0 --batch 2 --imgsz 640 --workers 0
```

Record each printed run directory. Both create a fresh timestamped folder.
Compare their `weights/last.pt` to the epoch 9 starting checkpoint using
`src/evaluate.py --split val` on `configs/yolo_dataset.yaml`, with matching IoU
0.50 and human precision, recall and false detections at 0.25 confidence.
Use the last checkpoints to compare after the same number of training epochs.
Use `scripts/probe_human_detection.py --sizes 640` for the same small-human
validation sample across all three checkpoints. Repeat the selected comparison
on real TestReal sequences after reviewing validation.

## Targeted human augmentation

The targeted human augmentation arm uses the existing `real_only` dataset,
epoch 9 starting weights, and the control's three-epoch budget. Add
`--human-augmentation` to the real-only command. The extra transform affects
257 of the 500 human-containing frames, selected deterministically by filename.
It flips those frames and their boxes, and applies a contrast factor of 0.9 or
1.1 with a brightness offset of minus or plus 6 on the 8-bit scale. It applies
before the standard training transforms. The new transform preserves box sizes
and leaves validation frames untouched. The resulting run is identifiable by
its `birdsai_human_augmentation_` prefix and its run manifest.

`scripts/evaluate_human_pilot.py` reproduces the fixed-threshold comparison,
including human boxes below 16 pixels on the 640-pixel input scale. Run it
with `--model` pointing to the augmented run's `last.pt` to compare three
epochs with the real-only control.

The completed augmentation run was
`runs/birdsai_human_augmentation_20261117_132933_718181`. It used CPU
non-maximum suppression during training validation after the first attempt hit
XPU NMS timeouts. The final comparison used XPU and the same fixed settings as
the saved pilot comparison. The epoch 3 checkpoint matched 1 of 2,697 human
boxes, with 126 false human detections and zero matches among 2,618 small
human boxes. The real-only control matched zero, with 119 false human
detections. The starting epoch 9 checkpoint matched 11, with 260 false human
detections. The augmented checkpoint did not improve small-human recall.

## Fine-tuning after feature extraction

The mixed thermal dataset is `data/finetune_pilot_v1/real_plus_hit_uav/dataset.yaml`.
It contains 1,500 training images and uses 3,098 BIRDSAI validation images.
The detector learns animal and human together. Starting from epoch 9,
`freeze=22` updates the detection head while keeping the earlier modules'
parameters frozen. Each experiment writes to its own run directory.

The completed 640-pixel experiment
`runs/birdsai_low_lr_freeze22_20261117_222406_925673` ran for three epochs.
At confidence 0.25 and matching IoU 0.50, its last checkpoint matched
50 of 2,697 validation human boxes, including 37 small boxes, with 818
false human detections. The epoch-9 baseline matched 11 human boxes with
260 false detections. Keep epoch 9 for the demo until a candidate passes
an evaluation of both classes and the false-alarm trade-off.

The later one-epoch 768-pixel attempt
`runs/birdsai_low_lr_freeze22_20261118_212039_333636` has settings saved
but no checkpoint or metrics. It cannot be compared with the baseline.

## Reduce background false alarms

The live demo accepts separate `--live-animal-conf` and `--live-human-conf`
thresholds. Both default to 0.25. Compare a higher animal threshold on
the same empty scenes and genuine animal clips before choosing a demo setting.
Track false boxes alongside missed animals and humans.

The next training dataset should add independent thermal human instances
across locations, distances and poses, and reviewed background frames
that trigger animal boxes. A verified empty frame has no object boxes.
Keep genuine human and animal labels in any nonempty frame. Draw training
negatives from training sequences or new footage reserved for training.
Keep validation and test sequences separate. Compare candidates with
epoch 9 using human recall, animal recall, and false alarms on empty footage.

## Object-centred thermal augmentation

`scripts/prepare_visible_thermal.py` retains the 1,500 mixed training
frames and adds 160 human-centred and 80 animal-centred crops sampled across
sequences. The crop targets the smallest box of the selected class in the
frame, retaining roughly 60% of each image dimension when the target fits.
Contrast is increased 10%, and all intersecting boxes are clipped to the
crop and renormalized. The original full frames preserve scene context.
These are additional training views of existing annotations.

The output dataset is `data/finetune_pilot_v1/real_plus_hit_uav_visible_85_15/dataset.yaml`.
It targets 1,740 training images and 307 validation images (about 85/15).
Validation frames are sampled across the existing held-out validation sequences
and copied without augmentation. The manifest records every crop, its source,
and the validation frame names. The full validation set remains available for
comparison with earlier runs. No test data is used or edited.
Preflight of the source labels found 1,282 animal boxes and 199 human boxes
in the planned 307 validation frames, with zero source sequence IDs shared
between training and validation. This check precedes the full crop export.
This variant does not add reviewed empty scenes. Compare its false alarms
as well as recall before using its checkpoint in the demo.

## Rebuild

`scripts/prepare_finetune_pilot.py` requires the original corrected BIRDSAI
dataset, the downloaded simulation sample, and the project's Python packages.
It publishes training YAMLs only after both arms pass validation. An existing
pilot output directory causes it to exit before writing.
