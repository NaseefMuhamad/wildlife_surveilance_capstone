"""Mix verified real HIT-UAV people with the retained BIRDSAI pilot frames."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_utils import file_hash, validate_root, write_json
from prepare_finetune_pilot import paired_images, select_across_sequences, export_pairs


def main():
    external = ROOT / "data/external/hit_uav_humans_v1"
    source = json.loads((external / "manifest.json").read_text())
    assert len(source["records"]) == 450
    control = ROOT / "data/finetune_pilot_v1/real_only"
    output = ROOT / "data/finetune_pilot_v1/real_plus_hit_uav"
    if output.exists():
        raise FileExistsError(output)
    humans, animals = [], []
    for image, label in paired_images(control, "train"):
        if any(row.startswith("1 ") for row in label.read_text().splitlines()):
            humans.append((image, label))
        else:
            animals.append((image, label))
    real_humans = select_across_sequences(humans, 150)
    training = animals + real_humans + paired_images(external, "train")
    assert len(training) == 1500
    validation = paired_images(control, "val")
    held_out = validation + paired_images(external, "val")
    test = paired_images(ROOT / "data/processed_v3_seed1", "test")
    hashes = {file_hash(image) for image, _ in held_out + test}
    if any(file_hash(image) in hashes for image, _ in training):
        raise ValueError("Image bytes shared with evaluation data")
    for split, pairs in (("train", training), ("val", validation)):
        export_pairs(output, split, pairs)
    inventory = validate_root(output, ("train", "val"), verify_images=False)
    manifest = json.loads((control / "manifest.json").read_text())
    if inventory["val"] != manifest["splits"]["val"]:
        raise ValueError("Real validation changed")
    write_json(output / "manifest.json", {
        "splits": inventory, "external_manifest_sha256": file_hash(external / "manifest.json"),
        "selection": "1000 real BIRDSAI animal frames, 150 real BIRDSAI human frames, 350 real HIT-UAV frames",
        "external_validation": str(external / "images/val"),
    })
    (output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: animal\n  1: human\n")
    print(json.dumps(inventory), flush=True)


if __name__ == "__main__":
    main()
