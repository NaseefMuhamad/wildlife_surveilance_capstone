"""Run a three-epoch, explicit low-learning-rate control on the real-only pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_utils import PROJECT, absolute_config, file_hash, write_json
from pipeline_cli import name_now, require_model, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--data", default="data/finetune_pilot_v1/real_only/dataset.yaml")
    args = parser.parse_args()
    if args.freeze < 0:
        parser.error("--freeze must be nonnegative")
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.imgsz % 32:
        parser.error("Use positive epochs and batch, and an image size divisible by 32.")
    _, YOLO = runtime("xpu:0")
    from ultralytics.utils import nms

    checkpoint = require_model("runs/birdsai_corrected_20261114_234448_294790/weights/best.pt")
    from dataset_utils import dataset_config
    dataset_root, _ = dataset_config(args.data)
    manifest = dataset_root / "manifest.json"
    original = nms.non_max_suppression

    def cpu_nms(prediction, *args, **kwargs):
        primary = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        if primary.device.type != "xpu":
            return original(prediction, *args, **kwargs)
        kwargs["max_time_img"] = 10.0
        return [item.to(primary.device) for item in original(primary.cpu(), *args, **kwargs)]

    def record(trainer):
        write_json(Path(trainer.save_dir) / "run_manifest.json", {
            "initial_checkpoint": str(checkpoint),
            "initial_checkpoint_sha256": file_hash(checkpoint),
            "dataset_manifest_sha256": file_hash(manifest),
            "optimizer": "AdamW", "lr0": 0.00005, "warmup_epochs": 0,
            "purpose": "Explicit low-learning-rate fine-tuning pilot",
            "freeze": args.freeze,
            "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        })

    model = YOLO(str(checkpoint))
    model.add_callback("on_pretrain_routine_end", record)
    nms.non_max_suppression = cpu_nms
    try:
        model.train(
            data=str(absolute_config(args.data)),
            device="xpu:0", epochs=args.epochs, patience=0, batch=args.batch, imgsz=args.imgsz, workers=0,
            amp=False, optimizer="AdamW", lr0=0.00005, lrf=0.1,
            warmup_epochs=0, seed=0, val=False, plots=True,
            freeze=args.freeze,
            project=str(PROJECT / "runs"), name=name_now(f"birdsai_low_lr_freeze{args.freeze}"),
        )
    finally:
        nms.non_max_suppression = original


if __name__ == "__main__":
    main()
