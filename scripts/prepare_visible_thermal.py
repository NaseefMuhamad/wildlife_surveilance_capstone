"""Add object-centred crops to the mixed thermal fine-tuning dataset."""
import math
from pathlib import Path

from PIL import Image, ImageEnhance

from human_tiles import load_boxes, tile_labels
from prepare_finetune_pilot import export_pairs, paired_images, select_across_sequences
from dataset_utils import file_hash, validate_root, write_json


ROOT = Path(__file__).resolve().parents[1] / "data/finetune_pilot_v1"
SOURCE = ROOT / "real_plus_hit_uav"
OUTPUT = ROOT / "real_plus_hit_uav_visible_85_15"


def main():
    training = paired_images(SOURCE, "train")
    human_frames, animal_frames = [], []
    for image, label in training:
        classes = {row.split()[0] for row in label.read_text(encoding="utf-8").splitlines() if row.strip()}
        if "1" in classes:
            human_frames.append((image, label))
        elif "0" in classes:
            animal_frames.append((image, label))
    groups = [(1, select_across_sequences(human_frames, 160)),
              (0, select_across_sequences(animal_frames, 80))]
    train_count = len(training) + sum(len(pairs) for _, pairs in groups)
    validation = select_across_sequences(
        paired_images(SOURCE, "val"), round(train_count * 15 / 85),
    )

    OUTPUT.mkdir(exist_ok=False)
    export_pairs(OUTPUT, "train", training)
    export_pairs(OUTPUT, "val", validation)
    crops = []
    for cls, pairs in groups:
        for image, label in pairs:
            with Image.open(image) as frame:
                width, height = frame.size
                boxes = load_boxes(label, width, height)
                _, left, top, right, bottom = min(
                    (box for box in boxes if box[0] == cls),
                    key=lambda box: (box[3] - box[1]) * (box[4] - box[2]),
                )
                crop_width = min(width, max(round(width * 0.6), math.ceil(right - left) + 4))
                crop_height = min(height, max(round(height * 0.6), math.ceil(bottom - top) + 4))
                x = max(0, min(width - crop_width, round((left + right - crop_width) / 2)))
                y = max(0, min(height - crop_height, round((top + bottom - crop_height) / 2)))
                region = (x, y, x + crop_width, y + crop_height)
                _, rows = tile_labels(boxes, region)
                if not any(row.startswith(f"{cls} ") for row in rows):
                    raise ValueError(f"Target lost in crop: {image.name}")
                sequence, frame_number = image.stem.rsplit("_", 1)
                stem = f"{sequence}_1{frame_number.zfill(12)}"
                output_image = OUTPUT / "images/train" / (stem + ".jpg")
                crop = ImageEnhance.Contrast(frame.convert("RGB").crop(region)).enhance(1.10)
                crop.save(output_image, quality=95)
                (OUTPUT / "labels/train" / (stem + ".txt")).write_text(
                    "\n".join(rows) + "\n", encoding="utf-8",
                )
                crops.append({"source": image.name, "image": output_image.name,
                              "target_class": cls, "region": region, "contrast": 1.10})
    inventory = validate_root(OUTPUT, ("train", "val"), verify_images=False)
    if set(inventory["train"]["sequences"]) & set(inventory["val"]["sequences"]):
        raise ValueError("Training and validation sequences overlap")
    for split, expected_count in (("train", train_count), ("val", len(validation))):
        if inventory[split]["images"] != expected_count:
            raise ValueError(f"{split}: unexpected image count")
        if not all(inventory[split]["boxes"].get(cls) for cls in ("human", "animal")):
            raise ValueError(f"{split}: both object classes must be represented")
    write_json(OUTPUT / "manifest.json", {
        "source_manifest_sha256": file_hash(SOURCE / "manifest.json"),
        "crops": crops, "splits": inventory,
        "validation_frames": [image.name for image, _ in validation],
    })
    (OUTPUT / "dataset.yaml").write_text((SOURCE / "dataset.yaml").read_text(encoding="utf-8"),
                                         encoding="utf-8")
    print({split: (data["images"], data["boxes"]) for split, data in inventory.items()})


if __name__ == "__main__":
    main()
