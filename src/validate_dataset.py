
import yaml
from pathlib import Path
import cv2

# Load the shared project configuration
with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

# First try the standard project location
images_root = Path(cfg["paths"]["train"]["images"])

# If the dataset is not there, try the local KaggleHub cache
if not images_root.exists():
    cache_root = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "manitagarwal"
        / "birdsai"
        / "versions"
        / "2"
    )
    cached_images = cache_root / "TrainReal" / "TrainReal" / "images"

    if cached_images.exists():
        images_root = cached_images
    else:
        raise FileNotFoundError(
            "Training images were not found.\n"
            f"Project path checked: {images_root}\n"
            f"KaggleHub path checked: {cached_images}"
        )

# Inspect images
sizes = {}
total_images = 0
unreadable_images = 0

for seq_folder in images_root.iterdir():
    if not seq_folder.is_dir():
        continue

    for img_file in seq_folder.glob("*.jpg"):
        img = cv2.imread(str(img_file))

        if img is None:
            unreadable_images += 1
            continue

        size = (img.shape[0], img.shape[1])
        sizes[size] = sizes.get(size, 0) + 1
        total_images += 1

print("\nImage Dimensions Found:\n")

for size, count in sorted(sizes.items()):
    print(f"{size}: {count} images")

print(f"\nTotal readable images: {total_images}")
print(f"Unreadable images: {unreadable_images}")
print(f"Images folder used: {images_root}")