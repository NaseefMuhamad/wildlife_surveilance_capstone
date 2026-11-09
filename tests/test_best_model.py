from ultralytics import YOLO

model = YOLO(
    r"runs/detect/runs/birdsai_yolov8n-2/weights/best.pt"
)

results = model.predict(
    source=r"data/raw/TestReal/TestReal/images/0000000362_0000000000",
    save=True,
    conf=0.25
)

print("Testing complete!")