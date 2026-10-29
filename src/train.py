from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="configs/yolo_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs",
    name="birdsai_yolov8n"
)