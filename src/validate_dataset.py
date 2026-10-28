import yaml
from pathlib import Path
import cv2

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

images_root = Path(cfg["paths"]["train"]["images"])

sizes = {}
total_images = 0

for seq_folder in images_root.iterdir():

    if not seq_folder.is_dir():
        continue

    for img_file in seq_folder.glob("*.jpg"):

        img = cv2.imread(str(img_file))

        if img is None:
            continue

        size = (img.shape[0], img.shape[1])

        sizes[size] = sizes.get(size, 0) + 1

        total_images += 1

print("\nImage Dimensions Found:\n")

for size, count in sorted(sizes.items()):
    print(f"{size}: {count}")

print(f"\nTotal Images: {total_images}")