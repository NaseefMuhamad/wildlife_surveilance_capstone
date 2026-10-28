# src/check_dataset_info.py

import yaml
from pathlib import Path
import cv2

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

images_root = Path(cfg["paths"]["train"]["images"])

first_sequence = next(images_root.iterdir())
first_image = next(first_sequence.glob("*.jpg"))

img = cv2.imread(str(first_image))

print("Sequence:", first_sequence.name)
print("Image:", first_image.name)
print("Shape:", img.shape)