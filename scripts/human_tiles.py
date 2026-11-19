"""Prepare a tile fine-tuning pilot and compare human detection on real validation frames."""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_utils import sequence_id, validate_root, write_json
from pipeline_cli import name_now, require_model, runtime
from prepare_finetune_pilot import paired_images
from probe_human_detection import match_boxes

CONTROL = ROOT / "data/finetune_pilot_v1/real_only"
PILOT = ROOT / "data/finetune_pilot_v1/real_tiles"


def regions(width, height):
    """Four tiles with 20% of the original frame shared between neighbors."""
    w, h = round(width * .6), round(height * .6)
    return [(x, y, x + w, y + h) for y in (0, height - h) for x in (0, width - w)]


def load_boxes(label, width, height):
    boxes = []
    for row in label.read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = map(float, row.split())
        boxes.append((int(cls), (x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height))
    return boxes


def tile_labels(boxes, region):
    x1, y1, x2, y2 = region
    w, h = x2 - x1, y2 - y1
    rows = []
    humans = 0
    for cls, left, top, right, bottom in boxes:
        a, b = max(0, left - x1), max(0, top - y1)
        c, d = min(w, right - x1), min(h, bottom - y1)
        if c <= a or d <= b:
            continue
        rows.append(f"{cls} {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}")
        if cls == 1 and (c-a)*(d-b) >= .8*(right-left)*(bottom-top):
            humans += 1
    return humans, rows


def copy_frame(source, target):
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def prepare():
    PILOT.mkdir(exist_ok=False)
    metadata = {}
    for split in ("train", "val"):
        images = PILOT / "images" / split
        labels = PILOT / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for image, label in paired_images(CONTROL, split):
            transform = split == "train" and int(image.stem.rsplit("_", 1)[1]) % 2 == 0
            if transform:
                with Image.open(image) as frame:
                    boxes = load_boxes(label, *frame.size)
                    choices = [(tile_labels(boxes, region), region) for region in regions(*frame.size)]
                    if any(box[0] == 1 for box in boxes):
                        (humans, rows), region = max(choices, key=lambda choice: choice[0][0])
                        transform = humans > 0
                    else:
                        (humans, rows), region = choices[int(image.stem.rsplit("_", 1)[1]) // 2 % 4]
                    if transform:
                        frame.crop(region).save(images / image.name, quality=95)
                        (labels / label.name).write_text("\n".join(rows), encoding="utf-8")
                        metadata[image.stem] = region
            if not transform:
                copy_frame(image, images / image.name)
                shutil.copy2(label, labels / label.name)
    inventory = validate_root(PILOT, ("train", "val"), verify_images=False)
    control = json.loads((CONTROL / "manifest.json").read_text(encoding="utf-8"))["splits"]
    if inventory["val"] != control["val"]:
        raise ValueError("Validation differs from the real-only pilot")
    write_json(PILOT / "manifest.json", {"splits": inventory, "cropped_frames": metadata})
    (PILOT / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: animal\n  1: human\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(metadata)} tiles and {inventory['train']['images'] - len(metadata)} full frames", flush=True)


def evaluate(model_path, stride, device, split):
    import cv2
    from torchvision.ops import nms
    from ultralytics.utils.metrics import box_iou

    torch, YOLO = runtime(device)
    model = YOLO(str(require_model(model_path)))
    if model.names != {0: "animal", 1: "human"}:
        raise ValueError("Expected the Reserve Watch animal/human checkpoint")
    counts = {mode: {threshold: Counter(tp=0, fp=0, small_tp=0)
                     for threshold in (.05, .10, .15, .25, .40, .60)}
              for mode in ("full", "tiles", "combined")}
    totals = Counter()
    sequences = Counter()
    dataset = CONTROL if split == "val" else ROOT / "data/processed_v3_seed1"
    started = perf_counter()
    for image, label in paired_images(dataset, split):
        sequence = sequence_id(image.stem)
        index = sequences[sequence]
        sequences[sequence] += 1
        if index % stride:
            continue
        frame = cv2.imread(str(image))
        height, width = frame.shape[:2]
        truth = [box[1:] for box in load_boxes(label, width, height) if box[0] == 1]
        small = [(box[3] - box[1]) * 640 / max(width, height) < 16 for box in truth]
        totals.update(frames=1, human_boxes=len(truth), small_human_boxes=sum(small))
        tiles = regions(width, height)
        inputs = [frame] + [frame[b:d, a:c] for a, b, c, d in tiles]
        predictions = model.predict(inputs, device=device, imgsz=640, conf=.05, iou=.7,
                                    classes=[1], verbose=False)
        candidates = []
        for result, (x, y, _, _) in zip(predictions, [(0, 0, width, height)] + tiles):
            boxes = result.boxes.xyxy.cpu() + torch.tensor([x, y, x, y])
            candidates.append((boxes, result.boxes.conf.cpu()))
        for mode, selection in (("full", candidates[:1]), ("tiles", candidates[1:]),
                                ("combined", candidates)):
            boxes = torch.cat([box for box, _ in selection])
            scores = torch.cat([score for _, score in selection])
            keep = nms(boxes, scores, .5) if mode != "full" else torch.arange(len(boxes))
            ious = box_iou(boxes, torch.tensor(truth).reshape(-1, 4)).numpy()
            for threshold, count in counts[mode].items():
                accepted = [i for i in keep.tolist() if scores[i] >= threshold]
                matches = match_boxes(ious, accepted, range(len(truth)), .5)
                count.update(tp=len(matches), fp=len(accepted) - len(matches),
                             small_tp=sum(small[i] for i in matches))
        if totals["frames"] % 100 == 0:
            print(f"Evaluated {totals['frames']} frames", flush=True)
    output = ROOT / "outputs/evaluation" / f"{name_now('human_tiles')}.json"
    write_json(output, {"checkpoint": str(model_path), "split": split, "stride": stride,
                        "seconds": round(perf_counter() - started, 1),
                        "match_iou": .5, "tile_nms_iou": .5, "totals": totals, "counts": counts})
    print(output, flush=True)
    print(json.dumps({"totals": totals, "counts": counts}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="xpu:0")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.model and args.stride > 0:
        evaluate(args.model, args.stride, args.device, args.split)
    else:
        parser.error("Specify --prepare or --model with a positive stride")
