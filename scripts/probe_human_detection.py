"""Compare inference settings on a systematic sample of validation frames."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline_cli import runtime


def match_boxes(ious, predictions, targets, minimum):
    candidates = sorted(
        (float(ious[box, target]), box, target)
        for box in predictions
        for target in targets
        if ious[box, target] >= minimum
    )
    matches = {}
    used_boxes = set()
    for _, box, target in reversed(candidates):
        if box not in used_boxes and target not in matches:
            matches[target] = box
            used_boxes.add(box)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 960])
    parser.add_argument("--device", default="xpu:0")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")

    torch, YOLO = runtime(args.device)
    import cv2
    from ultralytics.utils.metrics import box_iou

    dataset = ROOT / "data" / "processed_v3_seed1"
    grouped = defaultdict(list)
    for image in sorted((dataset / "images" / "val").glob("*.jpg")):
        grouped[image.stem.rsplit("_", 1)[0]].append(image)
    if not grouped:
        raise RuntimeError("No validation frames found")
    sample = [(sequence, image) for sequence, images in grouped.items() for image in images[::args.stride]]
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.25, 0.40]
    model = YOLO(str(args.model))
    report = {"checkpoint": str(args.model), "device": args.device, "stride": args.stride,
              "frames": len(sample), "matching_iou": 0.50, "sizes": {}}

    for size in args.sizes:
        sweep = {str(conf): Counter(tp=0, fp=0, fn=0) for conf in thresholds}
        audit = defaultdict(Counter)
        started = perf_counter()
        for sequence, image in sample:
            frame = cv2.imread(str(image))
            height, width = frame.shape[:2]
            label = dataset / "labels" / "val" / f"{image.stem}.txt"
            truth = []
            for row in label.read_text().splitlines():
                cls, x, y, w, h = map(float, row.split())
                if int(cls) == 1:
                    truth.append([(x - w / 2) * width, (y - h / 2) * height,
                                  (x + w / 2) * width, (y + h / 2) * height])
            result = model.predict(frame, conf=0.01, imgsz=size, iou=0.7,
                                   device=args.device, verbose=False)[0]
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            ious = box_iou(result.boxes.xyxy.cpu(), torch.tensor(truth).reshape(-1, 4)).numpy()
            targets = range(len(truth))
            human = [index for index, cls in enumerate(classes) if cls == 1]
            for conf in thresholds:
                accepted = [index for index in human if scores[index] >= conf]
                matched = match_boxes(ious, accepted, targets, 0.50)
                sweep[str(conf)].update(tp=len(matched), fp=len(accepted) - len(matched),
                                       fn=len(truth) - len(matched))
                if conf == 0.25:
                    default_matches = matched

            remaining = [target for target in targets if target not in default_matches]
            animals = [index for index, cls in enumerate(classes) if cls == 0 and scores[index] >= 0.25]
            wrong_class = match_boxes(ious, animals, remaining, 0.50)
            weak_humans = [index for index in human if scores[index] < 0.25]
            weak = match_boxes(ious, weak_humans, [t for t in remaining if t not in wrong_class], 0.50)
            for target, (_, top, _, bottom) in enumerate(truth):
                box_height = (bottom - top) * 640 / max(width, height)
                band = "under_16" if box_height < 16 else "16_to_31" if box_height < 32 else "32_plus"
                if target in default_matches:
                    status = "matched"
                elif target in wrong_class:
                    status = "animal_match"
                elif target in weak:
                    status = "weak_human_match"
                elif any(0.10 <= ious[index, target] < 0.50 for index in human):
                    status = "partial_human_overlap"
                else:
                    status = "other_miss"
                for key in ("all", f"sequence/{sequence}", f"height_at_640/{band}"):
                    audit[key][status] += 1

        report["sizes"][size] = {"sweep": sweep, "audit_at_0.25": audit,
                                 "seconds": round(perf_counter() - started, 1)}
        print(f"Finished {size}: {sweep}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
