"""Evaluate a pilot checkpoint at the fixed Reserve Watch validation threshold."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_utils import PROJECT, absolute_config, file_hash, write_json
from pipeline_cli import name_now, operating_validator, require_model, runtime
from probe_human_detection import match_boxes

CONFIDENCE = 0.25
MATCH_IOU = 0.50
IMAGE_SIZE = 640


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="xpu:0")
    args = parser.parse_args()
    checkpoint = require_model(args.model)
    _, YOLO = runtime(args.device)
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import box_iou

    original_nms = nms.non_max_suppression

    def relaxed_nms(*args, **kwargs):
        return original_nms(*args, **kwargs, max_time_img=10.0)

    class PilotValidator(operating_validator(CONFIDENCE, MATCH_IOU)):
        def init_metrics(self, model):
            super().init_metrics(model)
            matrix = self.confusion_matrix
            matrix.size_counts = Counter(small_total=0, small_matched=0)
            process = matrix.process_batch

            def measure(detections, batch, **kwargs):
                process(detections, batch, **kwargs)
                targets = batch["bboxes"][batch["cls"] == 1]
                if not len(targets):
                    return
                scale = IMAGE_SIZE / max(batch["ori_shape"]) / batch["ratio_pad"][0][0]
                small = ((targets[:, 3] - targets[:, 1]) * scale < 16).cpu().tolist()
                keep = (detections["cls"] == 1) & (detections["conf"] > CONFIDENCE)
                boxes = detections["bboxes"][keep]
                overlaps = box_iou(boxes.cpu(), targets.cpu()).numpy()
                matched = match_boxes(overlaps, range(len(boxes)), range(len(targets)), MATCH_IOU)
                matrix.size_counts.update(small_total=sum(small),
                                          small_matched=sum(small[target] for target in matched))

            matrix.process_batch = measure

    model = YOLO(str(checkpoint))
    if model.names != {0: "animal", 1: "human"}:
        raise ValueError("Expected an animal/human checkpoint")
    output = PROJECT / "outputs/evaluation" / name_now("human_pilot_fixed")
    nms.non_max_suppression = relaxed_nms
    try:
        metrics = model.val(data=str(absolute_config("configs/yolo_dataset.yaml")), split="val",
                            device=args.device, batch=8, imgsz=IMAGE_SIZE, workers=0, plots=True,
                            conf=CONFIDENCE, iou=0.7, validator=PilotValidator,
                            project=str(output.parent), name=output.name, exist_ok=False)
    finally:
        nms.non_max_suppression = original_nms
    matrix = metrics.confusion_matrix.matrix
    tp = int(matrix[1, 1])
    fp = int(matrix[1, :].sum()) - tp
    fn = int(matrix[:, 1].sum()) - tp
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "split": "val", "confidence": CONFIDENCE, "matching_iou": MATCH_IOU,
        "nms_iou": 0.7, "nms_max_seconds_per_image": 10, "batch": 8,
        "human_tp": tp, "human_fp": fp, "human_fn": fn,
        "human_precision": tp / max(1, tp + fp), "human_recall": tp / max(1, tp + fn),
        **metrics.confusion_matrix.size_counts,
    }
    write_json(output / "fixed_results.json", report)
    print(report, flush=True)
    print(f"Fixed-threshold results: {output}", flush=True)


if __name__ == "__main__":
    main()
