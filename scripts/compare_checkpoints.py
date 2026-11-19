"""Prepare a paired TestReal evaluation with optional telemetry disabled."""

import argparse
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "runs/birdsai_corrected_20261114_234448_294790/weights/best.pt",
    "runs/birdsai_low_lr_freeze22_20261117_222406_925673/weights/last.pt",
]


def block_python_network():
    """Reject Python socket connection and datagram events before any egress."""
    blocked = {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg"}

    def audit(event, arguments):
        if event in blocked:
            raise PermissionError(f"Comparison prohibits Python networking ({event})")

    sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs=2, default=DEFAULT_MODELS)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--check-network-guard", action="store_true")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.check_network_guard:
        block_python_network()
        import socket

        with socket.socket() as connection:
            try:
                connection.connect(("127.0.0.1", 1))
            except PermissionError:
                print("Python socket connection blocked before connect.")
                return 0
        raise RuntimeError("Python network guard failed")

    if args.device.startswith("xpu"):
        interpreter_file = PROJECT / ".xpu-python"
        if not interpreter_file.is_file():
            raise FileNotFoundError("Configure .xpu-python with the working Intel Arc Python interpreter")
        interpreter = Path(interpreter_file.read_text(encoding="utf-8").strip()).resolve()
        if not interpreter.is_file():
            raise FileNotFoundError(f"Configured GPU interpreter missing: {interpreter}")
        if Path(sys.executable).resolve() != interpreter:
            return subprocess.call([str(interpreter), "-E", str(Path(__file__).resolve()), *sys.argv[1:]],
                                   cwd=PROJECT)

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    block_python_network()
    sys.path.insert(0, str(PROJECT / "src"))
    sys.path.append(str(PROJECT / ".venv/Lib/site-packages"))

    import pipeline_cli
    from dataset_utils import validate_root

    # Decode dataset images once for the paired comparison.
    pipeline_cli.validate_root = lru_cache(None)(validate_root)
    for checkpoint in args.models:
        pipeline_cli.require_model(checkpoint)
    pipeline_cli.runtime(args.device)
    from ultralytics.utils import nms

    original_nms = nms.non_max_suppression
    original_write = pipeline_cli.write_json
    original_arguments = sys.argv

    def cpu_nms(prediction, *values, **options):
        primary = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        if primary.device.type != "xpu":
            return original_nms(prediction, *values, **options)
        options["max_time_img"] = 10.0
        return [item.to(primary.device)
                for item in original_nms(primary.cpu(), *values, **options)]

    def record(path, result):
        result.update(batch=args.batch, nms_backend="CPU for XPU input",
                      python_network_guard=True,
                      network_guard_scope="Python audit events only; native-library and host networking require separate verification")
        original_write(path, result)

    nms.non_max_suppression = cpu_nms
    pipeline_cli.write_json = record
    try:
        for checkpoint in args.models:
            sys.argv = ["evaluate.py", "--split", "test", "--device", args.device,
                        "--batch", str(args.batch), "--model", checkpoint,
                        "--cm-conf", "0.25", "--match-iou", "0.50"]
            pipeline_cli.evaluate_main()
    finally:
        nms.non_max_suppression = original_nms
        pipeline_cli.write_json = original_write
        sys.argv = original_arguments
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
