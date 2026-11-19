"""Score frozen Reserve Watch checkpoints on the date-held-out HIT-UAV sample."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_utils import PROJECT, file_hash, write_json
from human_tiles import load_boxes
from pipeline_cli import name_now, require_model, runtime
from prepare_finetune_pilot import paired_images
from probe_human_detection import match_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch, YOLO = runtime("xpu:0")
    import cv2
    from ultralytics.utils.metrics import box_iou

    checkpoint = require_model(args.model)
    model = YOLO(str(checkpoint))
    if model.names != {0: "animal", 1: "human"}:
        raise ValueError("Unexpected class names")
    counts = {str(conf): Counter(tp=0, fp=0, fn=0) for conf in (.10, .25, .50)}
    dataset = PROJECT / "data/external/hit_uav_humans_v1"
    for image, label in paired_images(dataset, "val"):
        frame = cv2.imread(str(image))
        height, width = frame.shape[:2]
        truth = [box[1:] for box in load_boxes(label, width, height)]
        pred = model.predict(frame, device="xpu:0", imgsz=640, conf=.10, iou=.7, verbose=False)[0]
        human = pred.boxes.cls == 1
        boxes, scores = pred.boxes.xyxy[human].cpu(), pred.boxes.conf[human].cpu().tolist()
        ious = box_iou(boxes, torch.tensor(truth).reshape(-1, 4)).numpy()
        for conf, count in counts.items():
            selected = [index for index, score in enumerate(scores) if score >= float(conf)]
            matches = match_boxes(ious, selected, range(len(truth)), .5)
            count.update(tp=len(matches), fp=len(selected)-len(matches), fn=len(truth)-len(matches))
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
              "source_manifest_sha256": file_hash(dataset / "manifest.json"),
              "split": "external dates held out", "matching_iou": .5, "counts": counts}
    path = PROJECT / "outputs/evaluation" / (name_now("hit_uav") + ".json")
    write_json(path, report)
    print(report, flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
