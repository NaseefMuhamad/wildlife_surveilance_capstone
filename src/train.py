from ultralytics import YOLO

model = YOLO(
    r"runs/detect/runs/birdsai_yolov8n-2/weights/last.pt"
)

model.train(resume=True)