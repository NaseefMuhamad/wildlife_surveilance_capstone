from pathlib import Path
from sklearn.model_selection import train_test_split
from shutil import move

# Current locations
img_dir = Path("data/processed/images/train")
lbl_dir = Path("data/processed/labels/train")

# Validation folders
val_img_dir = Path("data/processed/images/val")
val_lbl_dir = Path("data/processed/labels/val")

val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# Get all image names
images = sorted(img_dir.glob("*.jpg"))

# Split
train_imgs, val_imgs = train_test_split(
    images,
    test_size=0.15,
    random_state=42,
    shuffle=True
)

print(f"Train Images: {len(train_imgs)}")
print(f"Validation Images: {len(val_imgs)}")

# Move validation files
for img_path in val_imgs:

    label_path = lbl_dir / f"{img_path.stem}.txt"

    move(
        str(img_path),
        str(val_img_dir / img_path.name)
    )

    if label_path.exists():
        move(
            str(label_path),
            str(val_lbl_dir / label_path.name)
        )


print(
    "Train Images:",
    len(list(Path("data/processed/images/train").glob("*.jpg")))
)

print(
    "Val Images:",
    len(list(Path("data/processed/images/val").glob("*.jpg")))
)        

print("Validation split complete!")
