
from pathlib import Path
from collections import Counter
import random
import shutil

import cv2
import pandas as pd
import yaml


# --------------------------------------------------
# Configuration
# --------------------------------------------------

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

DATASET_ROOT = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "manitagarwal"
    / "birdsai"
    / "versions"
    / "2"
    / "TrainReal"
    / "TrainReal"
)

IMAGE_ROOT = DATASET_ROOT / "images"
ANNOTATION_ROOT = DATASET_ROOT / "annotations"

OUTPUT_ROOT = Path("data/processed_v2")
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

COLUMNS = cfg["annotation_columns"]
CLASS_IDS = {0, 1}


# --------------------------------------------------
# Safety checks
# --------------------------------------------------

if not IMAGE_ROOT.exists() or not ANNOTATION_ROOT.exists():
    raise FileNotFoundError(
        "The BIRDSAI training image or annotation folder was not found."
    )

if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise FileExistsError(
        f"{OUTPUT_ROOT} already contains files. "
        "Move or rename it before running this script again."
    )

csv_files = sorted(ANNOTATION_ROOT.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No annotation CSV files were found.")


# --------------------------------------------------
# Sequence-level split
# --------------------------------------------------

sequence_names = [
    csv_file.stem
    for csv_file in csv_files
    if (IMAGE_ROOT / csv_file.stem).is_dir()
]

if len(sequence_names) < 2:
    raise ValueError("Not enough valid sequences to create a split.")

random.Random(RANDOM_SEED).shuffle(sequence_names)

train_count = max(
    1,
    min(
        len(sequence_names) - 1,
        round(len(sequence_names) * TRAIN_RATIO),
    ),
)

train_sequences = set(sequence_names[:train_count])
val_sequences = set(sequence_names[train_count:])

print("Training sequences:", len(train_sequences))
print("Validation sequences:", len(val_sequences))


# --------------------------------------------------
# Create output folders
# --------------------------------------------------

for split in ("train", "val"):
    (OUTPUT_ROOT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT_ROOT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )


# --------------------------------------------------
# Convert annotations
# --------------------------------------------------

stats = Counter()
class_counts = Counter()
invalid_examples = []

for csv_file in csv_files:
    sequence_name = csv_file.stem
    image_folder = IMAGE_ROOT / sequence_name

    if not image_folder.is_dir():
        stats["missing_sequence_folders"] += 1
        continue

    split = (
        "train"
        if sequence_name in train_sequences
        else "val"
    )

    df = pd.read_csv(
        csv_file,
        header=None,
        names=COLUMNS,
    )

    images_by_frame = {}

    for image_path in image_folder.glob("*.jpg"):
        try:
            frame_number = int(image_path.stem.split("_")[-1])
        except ValueError:
            stats["invalid_image_filenames"] += 1
            continue

        if frame_number in images_by_frame:
            raise ValueError(
                f"Duplicate frame number {frame_number} "
                f"in sequence {sequence_name}"
            )

        images_by_frame[frame_number] = image_path

    # One label file for each annotated frame
    for frame_number, frame_df in df.groupby("frame_number"):
        try:
            frame_number = int(frame_number)
        except (ValueError, TypeError):
            stats["invalid_frame_numbers"] += len(frame_df)
            continue

        image_path = images_by_frame.get(frame_number)

        if image_path is None:
            stats["missing_annotated_frames"] += len(frame_df)
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            stats["unreadable_images"] += 1
            continue

        height, width = image.shape[:2]
        yolo_lines = []

        for _, row in frame_df.iterrows():
            try:
                cls = int(row["class"])
                x = float(row["x"])
                y = float(row["y"])
                box_w = float(row["w"])
                box_h = float(row["h"])
            except (ValueError, TypeError):
                stats["invalid_annotations"] += 1
                continue

            if cls not in CLASS_IDS:
                stats["unknown_classes"] += 1
                continue

            if (
                box_w <= 0
                or box_h <= 0
                or x < 0
                or y < 0
                or x + box_w > width
                or y + box_h > height
            ):
                stats["invalid_boxes"] += 1

                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        f"{sequence_name}, frame {frame_number}: "
                        f"box=({x}, {y}, {box_w}, {box_h}), "
                        f"image={width}x{height}"
                    )

                continue

            x_center = (x + box_w / 2) / width
            y_center = (y + box_h / 2) / height
            norm_w = box_w / width
            norm_h = box_h / height

            yolo_lines.append(
                f"{cls} {x_center:.6f} {y_center:.6f} "
                f"{norm_w:.6f} {norm_h:.6f}"
            )

            class_counts[cls] += 1
            stats["valid_boxes"] += 1

        # Include sequence name to prevent filename collisions
        output_stem = f"{sequence_name}_{frame_number:010d}"

        output_image = (
            OUTPUT_ROOT / "images" / split
            / f"{output_stem}.jpg"
        )

        output_label = (
            OUTPUT_ROOT / "labels" / split
            / f"{output_stem}.txt"
        )

        shutil.copy2(image_path, output_image)
        output_label.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8",
        )

        stats[f"{split}_images"] += 1
        stats[f"{split}_boxes"] += len(yolo_lines)

        if not yolo_lines:
            stats["empty_label_files"] += 1


# --------------------------------------------------
# Save split information and YOLO configuration
# --------------------------------------------------

(OUTPUT_ROOT / "train_sequences.txt").write_text(
    "\n".join(sorted(train_sequences)),
    encoding="utf-8",
)

(OUTPUT_ROOT / "val_sequences.txt").write_text(
    "\n".join(sorted(val_sequences)),
    encoding="utf-8",
)

dataset_yaml = (
    f"path: {OUTPUT_ROOT.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: animal\n"
    "  1: human\n"
)

(OUTPUT_ROOT / "dataset.yaml").write_text(
    dataset_yaml,
    encoding="utf-8",
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("\nCONVERSION SUMMARY")
print("-" * 50)

for key, value in sorted(stats.items()):
    print(f"{key}: {value}")

print("\nValid boxes by class:")
for cls, count in sorted(class_counts.items()):
    print(f"Class {cls}: {count}")

print("\nInvalid box examples:")
for example in invalid_examples:
    print(example)

print("\nOutput directory:", OUTPUT_ROOT.resolve())
print("Conversion finished.")
