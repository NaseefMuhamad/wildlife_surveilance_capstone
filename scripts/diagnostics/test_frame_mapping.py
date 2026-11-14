import pandas as pd
import yaml
from pathlib import Path

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

ann_root = Path(cfg["paths"]["train"]["annotations"])
img_root = Path(cfg["paths"]["train"]["images"])

csv_file = next(ann_root.glob("*.csv"))

seq_name = csv_file.stem

print("Sequence:", seq_name)

df = pd.read_csv(csv_file, header=None)

print("\nUnique frames:", df[0].nunique())

sequence_folder = img_root / seq_name

images = sorted(sequence_folder.glob("*.jpg"))

print("Images in sequence:", len(images))
print("First image:", images[0].name)
print("Last image:", images[-1].name)
