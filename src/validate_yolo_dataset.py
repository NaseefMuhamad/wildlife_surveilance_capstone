
from pathlib import Path
from collections import Counter

import yaml

root = Path("data/processed_v2")
errors = []
class_counts = Counter()
split_sequences = {}

for split in ("train", "val"):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    images = sorted(image_dir.glob("*.jpg"))
    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print(f"\n{split.upper()} SET")
    print("-" * 45)
    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    if missing_labels:
        errors.append(f"{split}: images without labels")
    if missing_images:
        errors.append(f"{split}: labels without images")

    # Sequence name is everything before the final frame number.
    split_sequences[split] = {
        p.stem.rsplit("_", 1)[0] for p in images
    }

    for label_path in labels:
        content = label_path.read_text(encoding="utf-8").strip()

        if not content:
            errors.append(f"Empty label: {label_path}")
            continue

        for line_number, line in enumerate(
            content.splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"Invalid format: {label_path}, line {line_number}"
                )
                continue

            try:
                cls = int(parts[0])
                x_center, y_center, width, height = map(
                    float, parts[1:]
                )
            except ValueError:
                errors.append(
                    f"Non-numeric values: {label_path}, "
                    f"line {line_number}"
                )
                continue

            if cls not in (0, 1):
                errors.append(
                    f"Unexpected class {cls}: {label_path}"
                )
                continue

            class_counts[cls] += 1

            if not all(
                0 <= value <= 1
                for value in (
                    x_center, y_center, width, height
                )
            ):
                errors.append(
                    f"Out-of-range coordinates: {label_path}, "
                    f"line {line_number}"
                )

overlap = (
    split_sequences["train"] & split_sequences["val"]
)

print("\nSEQUENCE SPLIT CHECK")
print("-" * 45)
print("Training sequences:", len(split_sequences["train"]))
print("Validation sequences:", len(split_sequences["val"]))
print("Overlapping sequences:", len(overlap))

if overlap:
    errors.append(f"Train/validation overlap: {sorted(overlap)}")

print("\nBOX COUNTS")
print("-" * 45)
print("Class 0:", class_counts[0])
print("Class 1:", class_counts[1])
print("Total boxes:", sum(class_counts.values()))

print("\nVALIDATION RESULT")
print("-" * 45)

if errors:
    print("CHECKS FAILED:", len(errors))
    for error in errors[:20]:
        print("-", error)
else:
    print("All structural checks passed.")
