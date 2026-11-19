"""Import a date-separated HIT-UAV human sample for Reserve Watch."""

import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/hit_uav_source"
OUTPUT = ROOT / "data/external/hit_uav_humans_v1"
BASE = "https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/main/"


def download(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=20) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise


def choose(images, count):
    groups = defaultdict(list)
    for image in images:
        groups[(image["date_captured"], image["filename"].rsplit("_", 1)[0])].append(image)
    for frames in groups.values():
        frames.sort(key=lambda image: image["filename"])
    quotas = dict.fromkeys(groups, 0)
    if len(images) < count:
        raise ValueError("Too few source frames")
    while count:
        for key, frames in sorted(groups.items()):
            if quotas[key] < len(frames):
                quotas[key] += 1
                count -= 1
                if count == 0:
                    break
    return [frames[(2 * index + 1) * len(frames) // (2 * quotas[key])]
            for key, frames in sorted(groups.items()) for index in range(quotas[key])]


def main():
    SOURCE.mkdir(parents=True, exist_ok=True)
    annotation_file = SOURCE / "train.json"
    if not annotation_file.exists():
        annotation_file.write_bytes(download(BASE + "normal_json/annotations/train.json"))
    data = json.loads(annotation_file.read_text())
    if {row["id"]: row["name"] for row in data["categories"]}.get(0) != "Person":
        raise ValueError("Unexpected source class mapping")
    labels = defaultdict(list)
    for row in data["annotation"]:
        if row["category_id"] == 0:
            labels[row["image_id"]].append(row)
    positive = [image for image in data["images"] if labels[image["id"]]]
    val_dates = {"20210114", "20210118"}
    selected = {
        "train": choose([image for image in positive if image["date_captured"] not in val_dates], 350),
        "val": choose([image for image in positive if image["date_captured"] in val_dates], 100),
    }
    for split in selected:
        for kind in ("images", "labels"):
            (OUTPUT / kind / split).mkdir(parents=True, exist_ok=True)

    def import_image(job):
        split, image = job
        path = OUTPUT / "images" / split / image["filename"]
        url = BASE + "normal_json/train/" + image["filename"]
        payload = path.read_bytes() if path.exists() else download(url)
        with Image.open(io.BytesIO(payload)) as frame:
            width, height = frame.size
            frame.verify()
        if (width, height) != (image["width"], image["height"]):
            raise ValueError(f"Image dimensions disagree for {path}")
        rows = []
        for box in labels[image["id"]]:
            x, y, w, h = box["bbox"]
            if w <= 0 or h <= 0 or min(x, y) < 0 or x+w > width or y+h > height:
                raise ValueError(f"Invalid human box for {path}")
            rows.append(f"1 {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}")
        if not path.exists():
            path.write_bytes(payload)
        (OUTPUT / "labels" / split / (path.stem + ".txt")).write_text("\n".join(rows), encoding="utf-8")
        return dict(split=split, filename=path.name, date=image["date_captured"],
                    source_url=url, sha256=hashlib.sha256(payload).hexdigest(), human_boxes=len(rows))

    jobs = [(split, image) for split, images in selected.items() for image in images]
    records = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for record in pool.map(import_image, jobs):
            records.append(record)
            if len(records) % 50 == 0:
                print(f"Verified {len(records)} of {len(jobs)} frames", flush=True)
    hashes = {split: {r["sha256"] for r in records if r["split"] == split} for split in selected}
    if hashes["train"] & hashes["val"]:
        raise ValueError("Identical image bytes across training and validation")
    manifest = dict(
        dataset="HIT-UAV", license="CC-BY-4.0",
        source="https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset",
        annotation_sha256=hashlib.sha256(annotation_file.read_bytes()).hexdigest(),
        scope="Human-only labels, source Person 0 remapped to Reserve Watch human 1. Vehicles ignored.",
        split_method="Validation dates 20210114 and 20210118 held out in full; remaining dates for training.",
        selection="Evenly spaced frames per capture date and filename metadata group",
        records=records,
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: animal\n  1: human\n",
        encoding="utf-8",
    )
    counts = {split: dict(images=sum(r["split"] == split for r in records),
                          human_boxes=sum(r["human_boxes"] for r in records if r["split"] == split),
                          dates=sorted({r["date"] for r in records if r["split"] == split}))
              for split in selected}
    print(json.dumps(counts), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
