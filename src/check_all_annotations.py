
from pathlib import Path
import pandas as pd

# Locate the training annotations in the KaggleHub cache
annotation_dir = (
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
    / "annotations"
)

columns = [
    "frame_number", "object_id", "x", "y", "w",
    "h", "class", "species", "occlusion", "noise"
]

csv_files = sorted(annotation_dir.glob("*.csv"))

print("SEQUENCE SUMMARY")
print("-" * 80)
print("CSV files found:", len(csv_files))

all_classes = set()
total_rows = 0

for csv_file in csv_files:
    df = pd.read_csv(csv_file, header=None, names=columns)

    total_rows += len(df)
    all_classes.update(df["class"].dropna().unique())

    print(
        f"{csv_file.stem}: "
        f"rows={len(df)}, "
        f"frames={df['frame_number'].min()}-"
        f"{df['frame_number'].max()}, "
        f"classes={sorted(df['class'].dropna().unique())}"
    )

print("-" * 80)
print("Total annotation rows:", total_rows)
print("All detection classes:", sorted(all_classes))
