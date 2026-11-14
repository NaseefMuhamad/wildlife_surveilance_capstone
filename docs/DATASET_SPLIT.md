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
