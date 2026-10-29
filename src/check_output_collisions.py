
from pathlib import Path
from collections import Counter

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
    / "images"
)

# Check whether frame-only filenames are repeated across sequences
frame_names = []

for sequence_folder in base.iterdir():
    if sequence_folder.is_dir():
        for image in sequence_folder.glob("*.jpg"):
            frame_names.append(image.name.split("_")[-1])

counts = Counter(frame_names)
duplicates = {
    name: count
    for name, count in counts.items()
    if count > 1
}

print("Total images:", len(frame_names))
print("Unique frame-only filenames:", len(counts))
print("Duplicated frame-only filenames:", len(duplicates))
print("Images involved in duplicate names:",
      sum(duplicates.values()))

print("\nExamples:")
for name, count in list(duplicates.items())[:10]:
    print(f"{name}: {count} occurrences")
