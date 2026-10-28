# src/check_processed_dataset.py

from pathlib import Path

images = len(list(Path("data/processed/images/train").glob("*.jpg")))
labels = len(list(Path("data/processed/labels/train").glob("*.txt")))

print(f"Images: {images}")
print(f"Labels: {labels}")

print("\nDifference:", images - labels)
