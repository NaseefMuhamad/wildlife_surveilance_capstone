
from pathlib import Path
import pandas as pd
import yaml

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

base = (
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

images_root = base / "images"
annotations_root = base / "annotations"
columns = cfg["annotation_columns"]

total_sequences = 0
total_annotations = 0
matched_annotations = 0
missing_frames = 0
missing_folders = 0

for csv_file in sorted(annotations_root.glob("*.csv")):
    total_sequences += 1
    sequence_name = csv_file.stem
    image_folder = images_root / sequence_name

    if not image_folder.exists():
        print(f"MISSING IMAGE FOLDER: {sequence_name}")
        missing_folders += 1
        continue

    # Read actual frame numbers from image filenames
    image_frames = set()

    for image_path in image_folder.glob("*.jpg"):
        try:
            image_frames.add(int(image_path.stem.split("_")[-1]))
        except ValueError:
            continue

    df = pd.read_csv(csv_file, header=None, names=columns)

    annotation_frames = set(
        pd.to_numeric(df["frame_number"], errors="coerce")
        .dropna()
        .astype(int)
    )

    matched = annotation_frames & image_frames
    missing = annotation_frames - image_frames

    total_annotations += len(df)
    matched_annotations += int(
        df["frame_number"].isin(matched).sum()
    )
    missing_frames += len(missing)

    print(
        f"{sequence_name}: "
        f"images={len(image_frames)}, "
        f"annotated frames={len(annotation_frames)}, "
        f"matched frames={len(matched)}, "
        f"missing frame IDs={len(missing)}"
    )

print("\n" + "=" * 65)
print("SUMMARY")
print("Sequences checked:", total_sequences)
print("Annotation rows:", total_annotations)
print("Annotation rows on matched frames:", matched_annotations)
print("Missing annotated frame IDs:", missing_frames)
print("Missing image folders:", missing_folders)
