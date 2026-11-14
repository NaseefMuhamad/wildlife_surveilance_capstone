# Thermal Wildlife Surveillance — corrected detection pipeline

This repository detects **animal (0)** and **human (1)** in BIRDSAI thermal imagery.
It does not identify ranger identities or classify people as threats. Tracking,
patrol registration, triage and a React dashboard are future application work.

## Install (Python 3.11, Windows)

Create a fresh environment so an existing training environment is preserved.
Choose ONE backend first, then install the shared requirements:

```powershell
py -3.11 -m venv .venv-xpu
.venv-xpu/Scripts/python.exe -m pip install -r requirements-xpu.txt
.venv-xpu/Scripts/python.exe -m pip install -r requirements.txt
```

For CPU, use `.venv-cpu` and `requirements-cpu.txt` instead. The Intel Arc setup
tested locally uses torch 2.8.0+xpu and torchvision 0.23.0+xpu. A generic latest
PyTorch install is not equivalent. Drivers must support the selected backend.
Commands below assume `python` selects the environment you installed.

## One preparation pipeline

Run from the project root (the scripts also resolve their own config paths):

```powershell
python src/prepare_yolo_dataset.py --seed 1 --output data/processed_v3_seed1 --allow-missing-frames
python src/validate_yolo_dataset.py --data configs/yolo_dataset.yaml
```

Default raw locations are in `configs/dataset.yml`. Use `--train-root` and
`--test-root` to override them; each root must contain `images/` and
`annotations/`. `--output` selects a NEW output directory. Existing datasets are
never overwritten or moved. Use the generated `dataset.yaml` for custom outputs.

With 32 TrainReal sequences, seed 1 and ratio 0.85 yield **27 training and 5
validation sequences**, with human and animal labels on both sides. TestReal stays
separate. Frame IDs are matched to filenames; gaps never become list indices.
The old converter delegates to this pipeline; the old frame-shuffle splitter
fails with instructions instead of moving files.

Missing annotated images are an error unless `--allow-missing-frames` is supplied.
The local TestReal audit found 87 missing annotated frame IDs. Exclusions, raw
annotation hashes, sequence lists, image/box counts and label hashes are recorded
in `manifest.json`. Invalid boxes are errors, not silently discarded labels.

Only frames with annotation rows are exported. Images without rows are not
assumed to be verified empty scenes, so the resulting evaluation is not a full
field false-alarm benchmark. Empty YOLO labels are supported for verified empty
scenes, but the current MOT conversion does not invent them.

For faster local preparation, `--link-images` can hardlink images rather than
copying them. Never edit those output image files: they share bytes with raw
images. Labels and manifests are always separate files.

## Read-only audit of a legacy dataset

```powershell
python src/audit_label_mapping.py --dataset data/processed
```

This checks deterministic first/middle/last samples per sequence/split against
the raw images and same-frame CSV boxes. It is a spot audit, not a certification
of every annotation. `docs/original_label_audit.json` contains the local audit.

## Train and resume explicitly

Start a NEW run from original COCO pretrained weights after rebuilding the split:

```powershell
python src/train.py --data configs/yolo_dataset.yaml --model yolov8n.pt --device xpu:0 --batch 2 --epochs 50
```

Use `--device cpu` for CPU. Supply an existing, nonempty local checkpoint;
automatic weight downloads and optional Ultralytics analytics are disabled by
the entry points. Fresh training uses full precision (`amp=False`) because AMP
failed the compatibility check on the locally tested Intel setup.

The old trained `best.pt` is preserved for inference and frozen TestReal testing.
Do not use it as initialization and then claim the rebuilt validation split is
unseen: it may already have learned from those sequences. Code corrections do
not retroactively correct a model's training history or validation scores.

Resume a corrected run with `--resume path/to/last.pt`. It must have a matching
`run_manifest.json` and dataset manifest. Legacy resume without provenance is
rejected. Workers default to zero; training has a Windows multiprocessing guard.

## Predict a local image or sequence (CPU default)

```powershell
python src/detect.py --source data/raw/TestReal/TestReal/images/image.png --device cpu --conf 0.25
```

The default checkpoint is the existing fine-tuned
`runs/detect/runs/birdsai_yolov8n-2/weights/best.pt`. Override it with `--model`.
For GPU use `--device xpu:0` in the compatible environment. The COCO baseline
requires an explicit `--model yolov8n.pt`; its classes differ from this task.
`tests/test_best_model.py` is a compatible entry point with the same arguments.
Sequence prediction streams results rather than retaining them all in memory.

Annotated images, `detections.jsonl` and checkpoint/device metadata go to a new
`outputs/predictions/prediction_<timestamp>/` directory. Zero detections means
none passed the selected threshold; it is not a runtime error. CPU selection
does not solve a domain/viewpoint mismatch in an internet image.

## Evaluate a fixed checkpoint

```powershell
python src/evaluate.py --data configs/yolo_dataset.yaml --split test --device xpu:0 --batch 4 --cm-conf 0.25 --match-iou 0.50
```

CPU works with `--device cpu`. Use `--model` for another fine-tuned two-class
checkpoint. Save and report the chosen checkpoint hash; do not tune on TestReal.
Results go to a new `outputs/evaluation/test_<timestamp>/` folder.

AP uses a confidence floor of **0.001** and standard IoUs 0.50–0.95. The
confusion matrix separately uses **confidence 0.25 and matching IoU 0.50** by
default. Matrix filenames include those thresholds. NMS IoU is a separate 0.70
setting. `test_results.json` includes both AP and fixed-threshold class metrics,
the matrix, data/checkpoint hashes, inventory, backend and timing.

Standard training-generated confusion matrices in this installed Ultralytics
version use the AP confidence floor (0.001). Many low-confidence false positives
can coexist with high AP; do not read those old plots as performance at 0.25.
Use this evaluation command for a documented operating-point matrix.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Regression coverage includes gapped frame IDs, the 27/5 split, leakage, missing
datasets, invalid/zero-size/nonfinite boxes, missing pairs, changed manifests,
explicit missing-frame exclusions and fixed confusion-matrix thresholds.

The existing progress report is retained as historical documentation. Read
`docs/CORRECTIONS.md` for the audit findings and what has actually been verified.

## Ranger triage demo

The Python API lives at `backend/triage_demo.py`. Training and inference code
lives in `src/`. See [Project roadmap](docs/ROADMAP.md) for the repository
layout, demo setup, and paths to retained results.
From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m backend.triage_demo
```

Start the standalone dashboard repository on port 5173 and open
http://127.0.0.1:5173 while both servers run. Press
**Play replay**, check the patrol register when an alert opens, and record a
review outcome. **Clear demo reviews** clears alerts and keeps the patrol roster.
Use **Layout** to preview Camera wall, North Trail focus, or Ranger review.
The original dashboard is preserved in `demos/coursework/` and runs via
`.\.venv\Scripts\python.exe src\triage_demo.py` on port 8765.

The backend uses saved BIRDSAI inference and stores patrols and reviews in
`outputs/triage_demo/reviews.json`. Patrol assignments and replay timestamps
are simulated. Use `--run` to point to another saved prediction directory
containing `detections.jsonl` and annotated frames, or `--port` if 8765 is busy.

### Object tracking

Add `--track` to `src/detect.py` when processing one ordered image sequence or
video. Each visible detection in `detections.jsonl` then includes a `track_id`.
The dashboard also assigns IDs when it replays saved detections. IDs are local
to a camera sequence and can change after a long occlusion or missed detection.
