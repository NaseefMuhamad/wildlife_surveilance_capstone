from ultralytics import YOLO

# Load pretrained YOLOv8
model = YOLO("yolov8n.pt")

# Run prediction on one thermal frame
results = model.predict(
    source=r"data/raw/TrainReal/TrainReal/images/0000000067_0000000041/0000000067_0000000041_0000000437.jpg",
    save=True,
    conf=0.25
)

print("Detection complete!")