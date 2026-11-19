"""Diagnostic memorization check on 16 real human frames, with intentional overlap."""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_utils import PROJECT, file_hash, write_json
from pipeline_cli import name_now, require_model, runtime
from prepare_finetune_pilot import paired_images
from probe_human_detection import match_boxes


def main():
    torch, YOLO = runtime("xpu:0")
    import cv2
    import numpy as np
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import box_iou

    output = PROJECT / "outputs/diagnostics" / name_now("human_memorization")
    dataset = output / "dataset"
    groups = defaultdict(list)
    for image, label in paired_images(PROJECT / "data/finetune_pilot_v1/real_only", "train"):
        people = [list(map(float, row.split())) for row in label.read_text().splitlines()
                  if row.startswith("1 ")]
        if people:
            groups[image.stem.rsplit("_", 1)[0]].append((min(row[4] for row in people), image, label))
    selected = []
    for sequence, frames in sorted(groups.items()):
        frames.sort(key=lambda item: (-item[0], item[1].name))
        selected.extend(frames[index] for index in range(min(4, len(frames))))
    if len(selected) != 16:
        raise ValueError(f"Expected four human sequences and 16 frames, got {len(selected)}")
    for kind in ("images", "labels"):
        (dataset / kind / "train").mkdir(parents=True)
    pairs = []
    thumbnails = []
    for _, image, label in selected:
        target = dataset / "images/train" / image.name
        shutil.copy2(image, target)
        shutil.copy2(label, dataset / "labels/train" / label.name)
        pairs.append((target, label))
        frame = cv2.imread(str(image))
        height, width = frame.shape[:2]
        for row in label.read_text().splitlines():
            cls, x, y, w, h = map(float, row.split())
            if cls == 1:
                cv2.rectangle(frame, (round((x-w/2)*width), round((y-h/2)*height)),
                              (round((x+w/2)*width), round((y+h/2)*height)), (0, 255, 0), 2)
        thumbnails.append(cv2.resize(frame, (320, 240)))
    cv2.imwrite(str(output / "labels_preview.jpg"), np.vstack(
        [np.hstack(thumbnails[index:index+4]) for index in range(0, 16, 4)]))
    config = dataset / "dataset.yaml"
    config.write_text(f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: animal\n  1: human\n")
    checkpoint = require_model("runs/birdsai_corrected_20261114_234448_294790/weights/best.pt")
    model = YOLO(str(checkpoint))

    def evaluate(detector):
        counts = {"human_boxes": 0, "matched": 0, "false_human_boxes": 0}
        for image, label in pairs:
            frame = cv2.imread(str(image))
            height, width = frame.shape[:2]
            truth = []
            for row in label.read_text().splitlines():
                cls, x, y, w, h = map(float, row.split())
                if cls == 1:
                    truth.append([(x-w/2)*width, (y-h/2)*height,
                                  (x+w/2)*width, (y+h/2)*height])
            result = detector.predict(frame, device="xpu:0", conf=.25, imgsz=640, verbose=False)[0]
            boxes = result.boxes.xyxy[result.boxes.cls == 1].cpu()
            overlaps = box_iou(boxes, torch.tensor(truth).reshape(-1, 4)).numpy()
            matches = match_boxes(overlaps, range(len(boxes)), range(len(truth)), .5)
            counts["human_boxes"] += len(truth)
            counts["matched"] += len(matches)
            counts["false_human_boxes"] += len(boxes) - len(matches)
        return counts

    original = nms.non_max_suppression

    def cpu_nms(prediction, *args, **kwargs):
        primary = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        if primary.device.type != "xpu":
            return original(prediction, *args, **kwargs)
        kwargs["max_time_img"] = 10.0
        return [item.to(primary.device) for item in original(primary.cpu(), *args, **kwargs)]

    nms.non_max_suppression = cpu_nms
    try:
        before = evaluate(model)
        print("BEFORE", before, flush=True)
        write_json(output / "diagnostic_manifest.json", {
            "scope": "Intentional train/evaluation overlap. Memorization test only.",
            "checkpoint_sha256": file_hash(checkpoint),
            "frames": [image.name for image, _ in pairs], "before": before,
        })
        model.train(data=str(config), device="xpu:0", epochs=100, patience=0,
                    batch=2, workers=0, imgsz=640, amp=False,
                    optimizer="AdamW", lr0=.001, lrf=1, warmup_epochs=0,
                    mosaic=0, mixup=0, scale=0, translate=0, fliplr=0,
                    hsv_h=0, hsv_s=0, hsv_v=0, auto_augment=None,
                    val=False, plots=False, save=True, seed=0,
                    project=str(output), name="training")
        after = evaluate(YOLO(str(output / "training/weights/last.pt")))
        write_json(output / "sanity_results.json", {
            "scope": "Memorization on the same 16 training frames, not generalization",
            "confidence": .25, "matching_iou": .5, "before": before, "after": after,
        })
        print("AFTER", after, flush=True)
        print(output, flush=True)
    finally:
        nms.non_max_suppression = original


if __name__ == "__main__":
    main()
