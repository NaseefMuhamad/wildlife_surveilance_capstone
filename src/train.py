import torch
from ultralytics import YOLO

# Load your checkpoint and run training safely on the CPU
model = YOLO(r"runs/detect/runs/birdsai_yolov8n-2/weights/last.pt")

model.train(
    resume=True,
    device="cpu",     # Switches to stable multi-core execution
    amp=True,         # Safe to re-enable on CPU
    plots=True,       # Safe to re-enable on CPU
    workers=4,        # Multi-threading works reliably on CPU
    batch=16          # Standard batch size for CPU stability
)