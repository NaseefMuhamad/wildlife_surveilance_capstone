# BIRDSAI dataset split

## Why the export has a new folder

`data/processed_v3_seed1` identifies the third dataset revision and records
the shuffle seed in the folder name. The original `processed` export had all
32 sequence IDs in both training and validation. The `processed_v2` export
removed sequence overlap, but its validation set contained no human boxes.
The new folder preserves those earlier exports for comparison with historical
results. `processed_v3` is the interrupted export. Training and evaluation now
use `processed_v3_seed1` through `configs/yolo_dataset.yaml`.

## How we split the dataset

1. Use the 32 TrainReal annotation CSV filenames to identify the sequences.
2. Sort the sequence IDs and shuffle them with Python's seeded random generator.
   Assign the first 27 sequences to training and the remaining five to validation.
   The target of 85% applies to sequences, so the image and box proportions differ.
3. Select seed 1 using TrainReal annotation counts. It gives both classes to
   training and validation and places human boxes in two validation sequences.
   This was a deterministic shuffle screened for class coverage.
4. Keep all 16 TestReal sequences separate. Match annotations to images by frame
   ID and export each pair using its sequence ID and frame ID. Convert boxes to
   normalized YOLO coordinates using the source image dimensions.

Keeping a whole sequence in one split prevents neighboring frames from the same
recording from appearing on both sides. Including both classes in validation
allows us to measure human and animal detection and choose operating thresholds
before using TestReal for the final comparison.

## Verified inventory

| Split | Sequences | Image-label pairs | Animal boxes | Human boxes |
|---|---:|---:|---:|---:|
| Training | 27 | 18,111 | 62,155 | 9,834 |
| Validation | 5 | 3,098 | 12,513 | 2,697 |
| TestReal | 16 | 15,494 | 56,908 | 21,853 |

Every exported image has a matching label. The audit found no duplicate image
stems within a split and no shared sequence IDs or byte-identical images across
splits. Repeated image content within training and within TestReal was retained.
Both classes remain imbalanced. Validation includes 2,697 human boxes, which
are frame-level annotations rather than a count of distinct people.

The exporter omitted 87 annotations' frame IDs with missing source images, all
in TestReal sequence `0000000011_0000000000`. The manifest lists each excluded
frame. Unannotated source images were excluded because their status as empty
scenes was unverified. This limits what the export can tell us about false alerts
on empty footage.

## Reproduction and evidence

From the repository root, use an unused output directory to reproduce the
split. The original export used hardlinks with copy fallback. Treat linked
images as read-only because they can share file contents with the raw dataset.

```powershell
python src/prepare_yolo_dataset.py --seed 1 --train-ratio 0.85 --output data/processed_v3_seed1_reproduction --link-images --allow-missing-frames
python src/validate_yolo_dataset.py --data data/processed_v3_seed1_reproduction/dataset.yaml
```

The original export's `manifest.json` and `*_sequences.txt` files record the
split membership, counts, label hashes and exclusions under
`data/processed_v3_seed1/`. The preparation code is
`src/prepare_yolo_dataset.py`. This dataset audit precedes retraining.

## Retraining preflight

The checked-in training YAML points to the dataset relative to the repository.
The exported dataset YAML uses `path: .`, which the project CLI resolves
relative to that YAML. The CLI supplies Ultralytics with an absolute path
resolved on the current machine. Absolute locations in manifests record
provenance. Reproduce the dataset from seed 1 and retain the manifest and
checkpoint hashes when comparing runs.

The training YAML resolves to this export, and the local pretrained
`yolov8n.pt` and historical `best.pt` were found. The active environment
reported PyTorch 2.15.0+xpu, torchvision 0.29.2+xpu and Ultralytics 8.4.183.
Intel Arc Graphics is available as `xpu:0`. A synthetic YOLOv8n batch of
two 640 × 640 images completed a forward pass, backward pass and SGD step
on that device. A full training epoch will establish the runtime on real data.

### Intel Arc training restart

PyTorch 2.15.0+xpu failed on the first real batch with a Level Zero
`UR_RESULT_ERROR_OUT_OF_RESOURCES` error during YOLO target preparation.
The synthetic preflight had not exercised that code path. We restarted
using the existing PyTorch 2.8.0+xpu environment from the earlier successful
Intel Arc run. The new run, `birdsai_corrected_20261114_233917_200688`,
passed the first real batches at 640 px, batch 2, workers 0 and AMP disabled.
It uses fresh `yolov8n.pt` weights, a maximum of 50 epochs and patience 10.
The assistant-started run was stopped on request during epoch 1, after
228 of 9,056 batches were logged. The user will launch the next run manually.

The manual command using `PYTHONPATH` loaded PyTorch from the project's
`.venv` before the GPU environment and failed to load `c10_xpu.dll`.
The training launcher now reads the local, Git-ignored `.xpu-python` file,
starts that interpreter with `-E`, and appends the project dependencies
after the GPU environment's packages. Run `python src/train.py` from the
repository root. The defaults are Intel Arc (`xpu:0`), 50 epochs and patience 10.
On another machine, use a compatible XPU environment or configure its
interpreter in that machine's own `.xpu-python` file.
