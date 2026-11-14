# src/check_image_sizes.py

import yaml
from pathlib import Path
import cv2

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

images_root = Path(cfg["paths"]["train"]["images"])

sizes = set()
count = 0

for seq_folder in images_root.iterdir():
    if not seq_folder.is_dir():
        continue

    for img_file in seq_folder.glob("*.jpg"):
        img = cv2.imread(str(img_file))

        if img is not None:
            sizes.add((img.shape[0], img.shape[1]))
            count += 1

        # stop after reasonable sample
        if count >= 1000:
            break

    if count >= 1000:
        break

print("Unique image sizes found:")
for s in sorted(sizes):
    print(s)

print(f"\nChecked {count} images")