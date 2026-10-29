from pathlib import Path

train_imgs = len(list(Path("data/processed/images/train").glob("*.jpg")))
train_lbls = len(list(Path("data/processed/labels/train").glob("*.txt")))

val_imgs = len(list(Path("data/processed/images/val").glob("*.jpg")))
val_lbls = len(list(Path("data/processed/labels/val").glob("*.txt")))

print("TRAIN")
print("Images:", train_imgs)
print("Labels:", train_lbls)

print("\nVAL")
print("Images:", val_imgs)
print("Labels:", val_lbls)