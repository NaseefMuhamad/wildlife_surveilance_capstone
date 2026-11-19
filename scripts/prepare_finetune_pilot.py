"""Build matched real-only and real-plus-simulation training datasets."""

import json
import shutil
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_utils import IMAGE_SUFFIXES, file_hash, sequence_id, validate_root, write_json

REAL = ROOT / "data/processed_v3_seed1"
SIM = ROOT / "data/external/birdsai_sim_human_sample"
OUTPUT = ROOT / "data/finetune_pilot_v1"


def paired_images(root, split=""):
    images = {p.stem: p for p in sorted((root / "images" / split).iterdir())
              if p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
    if images.keys() != labels.keys():
        raise ValueError(f"Mismatched images and labels in {root / split}")
    return [(image, labels[stem]) for stem, image in images.items()]


def select_across_sequences(pairs, count):
    groups = defaultdict(list)
    for pair in pairs:
        groups[sequence_id(pair[0].stem)].append(pair)
    if len(pairs) < count:
        raise ValueError(f"Need {count} frames, found {len(pairs)}")
    quotas = dict.fromkeys(groups, 0)
    while count:
        for sequence, frames in groups.items():
            if quotas[sequence] < len(frames):
                quotas[sequence] += 1
                count -= 1
                if count == 0:
                    break
    selected = []
    for sequence, frames in groups.items():
        quota = quotas[sequence]
        selected.append([frames[(2 * index + 1) * len(frames) // (2 * quota)]
                         for index in range(quota)])
    return [pair for row in zip_longest(*selected) for pair in row if pair is not None]


def export_pairs(destination, split, pairs):
    images = destination / "images" / split
    labels = destination / "labels" / split
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for image, label in pairs:
        try:
            (images / image.name).hardlink_to(image)
        except OSError:
            shutil.copy2(image, images / image.name)
        shutil.copy2(label, labels / label.name)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Pilot already exists at {OUTPUT}")
    real_manifest = json.loads((REAL / "manifest.json").read_text(encoding="utf-8"))
    synthetic_manifest = json.loads((SIM / "manifest.json").read_text(encoding="utf-8"))
    sim_hashes = {f"{row['sequence']}_{row['frame']:010d}": row["sha256"]
                  for row in synthetic_manifest["frames"]}
    simulation = paired_images(SIM)
    if {image.stem: file_hash(image) for image, _ in simulation} != sim_hashes or len(simulation) != 335:
        raise ValueError("Simulation sample differs from its verified manifest")

    humans, animals = [], []
    for image, label in paired_images(REAL, "train"):
        rows = label.read_text(encoding="utf-8").splitlines()
        if any(row.startswith("1 ") for row in rows):
            humans.append((image, label))
        elif rows:
            animals.append((image, label))

    selected_animals = select_across_sequences(animals, 1000)
    shared = select_across_sequences(humans, 500) + selected_animals[:665]
    validation = paired_images(REAL, "val")
    held_out_hashes = {file_hash(image) for image, _ in validation + paired_images(REAL, "test")}
    candidates = shared + selected_animals[665:] + simulation
    if any(file_hash(image) in held_out_hashes for image, _ in candidates):
        raise ValueError("An identical image appears in training and held-out footage")

    for name, extra in (("real_only", selected_animals[665:]),
                        ("real_plus_sim", simulation)):
        destination = OUTPUT / name
        training = shared + extra
        if len({image.stem for image, _ in training}) != 1500:
            raise ValueError(f"{name}: unexpected train size or duplicate image names")
        export_pairs(destination, "train", training)
        export_pairs(destination, "val", validation)
        inventory = validate_root(destination, ("train", "val"), verify_images=False)
        for key in ("images", "sequences", "label_sha256"):
            if inventory["val"][key] != real_manifest["splits"]["val"][key]:
                raise ValueError(f"{name}: validation {key} differs from the original export")
        write_json(destination / "manifest.json", {
            "arm": name,
            "source_manifest_sha256": file_hash(REAL / "manifest.json"),
            "simulation_manifest_sha256": file_hash(SIM / "manifest.json") if name == "real_plus_sim" else None,
            "shared_real_frames": len(shared),
            "training_frames": [image.stem for image, _ in training],
            "splits": inventory,
        })
        print(name, {split: (entry["images"], entry["boxes"]) for split, entry in inventory.items()}, flush=True)
    for name in ("real_only", "real_plus_sim"):
        (OUTPUT / name / "dataset.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: animal\n  1: human\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
