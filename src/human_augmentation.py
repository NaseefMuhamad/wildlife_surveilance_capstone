"""Optional training-only augmentation of real frames containing people."""

from hashlib import blake2b
from pathlib import Path

import numpy as np
from ultralytics.models.yolo.detect.train import DetectionTrainer


def augment_human_frame(sample):
    if not (sample["cls"] == 1).any():
        return sample

    name = Path(sample["im_file"]).name
    choice = blake2b(name.encode(), digest_size=1).digest()[0] % 4
    if choice < 2:
        return sample

    image = sample["img"]
    contrast, brightness = (0.9, -6) if choice == 2 else (1.1, 6)
    pixels = (image[:, ::-1].astype(np.float32) - 127.5) * contrast + 127.5 + brightness
    sample["img"] = np.clip(pixels, 0, 255).astype(np.uint8)
    instances = sample["instances"]
    instances.fliplr(1 if instances.normalized else image.shape[1])
    return sample


class HumanAugmentationTrainer(DetectionTrainer):
    """Add the targeted transform before the standard training transforms."""

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            original_loader = dataset.get_image_and_label

            def load(index):
                return augment_human_frame(original_loader(index))

            dataset.get_image_and_label = load
        return dataset
