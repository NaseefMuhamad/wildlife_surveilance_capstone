"""Phone HTTP camera capture. Model loading happens only when a camera connects."""
import os
import time
from threading import Event, Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4


COCO_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


class LiveCameras:
    def __init__(self, model_path, device, thresholds=None):
        self.model_path = model_path
        self.device = device
        self.thresholds = thresholds
        self.lock = Lock()
        self.cameras = {}

    def connect(self, source, name, location=""):
        source = validate_source(source)
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError("Enter a camera name of up to 60 characters.")
        if not isinstance(location, str) or len(location) > 60:
            raise ValueError("Enter a location of up to 60 characters.")
        with self.lock:
            if any(address == source for address, _, _, _ in self.cameras.values()):
                raise ValueError("That stream is already added. Remove its tile before reconnecting.")
            if len(self.cameras) >= 8:
                raise ValueError("The wall supports up to eight live cameras. Remove a tile to add another.")
            camera_id = uuid4().hex
            camera = LiveCamera(self.model_path, self.device, self.thresholds)
            camera.start(source)
            self.cameras[camera_id] = (source, name.strip(), location.strip(), camera)
            return camera_id

    def snapshots(self):
        with self.lock:
            return [{"id": key, "name": name, "location": location, **camera.snapshot()[0]}
                    for key, (_, name, location, camera) in self.cameras.items()]

    def frame(self, camera_id):
        with self.lock:
            entry = self.cameras.get(camera_id)
            return entry[3].snapshot() if entry else ({}, None)

    def disconnect(self, camera_id):
        if not isinstance(camera_id, str):
            raise ValueError("Choose a camera to disconnect.")
        with self.lock:
            entry = self.cameras.pop(camera_id, None)
        if entry:
            entry[3].stop()

    def stop(self):
        with self.lock:
            camera_ids = list(self.cameras)
        for camera_id in camera_ids:
            self.disconnect(camera_id)


def validate_source(source):
    if not isinstance(source, str) or len(source) > 2048:
        raise ValueError("Enter an HTTP camera stream URL.")
    parsed = urlsplit(source)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use an HTTP/MJPEG URL without embedded passwords.")
    return source


class LiveCamera:
    def __init__(self, model_path, device="cpu", thresholds=None):
        self.model_path, self.device = model_path, device
        self.thresholds = dict(thresholds or {"animal": 0.25, "human": 0.25})
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None
        self.jpeg = None
        self.state = {"status": "disconnected", "detections": [], "frames": 0}

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            if state.get("updated") and time.time() - state["updated"] > 10 and state["status"] == "live":
                state["status"] = "stalled"
            return state, self.jpeg

    def start(self, source):
        source = validate_source(source)
        self.stop()
        if self.thread and self.thread.is_alive():
            raise ValueError("Previous connection is closing. Try again in a few seconds.")
        self.stop_event.clear()
        with self.lock:
            self.jpeg = None
            self.state = {"status": "connecting", "detections": [], "frames": 0}
        self.thread = Thread(target=self._run, args=(source,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=0.2)
        with self.lock:
            self.jpeg = None
            self.state = {"status": "disconnected", "detections": [], "frames": 0}

    def _run(self, source):
        capture = None
        try:
            import cv2
            from src.tracking import DetectionTracker
            model = None
            model_error = None
            tracker = DetectionTracker()
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
            if not capture.isOpened():
                raise RuntimeError("Cannot open camera. Check Wi-Fi and the camera stream URL.")
            frames = 0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stream disconnected. Reconnect after checking the phone.")
                detections = []
                boxes = []
                if frames and self.model_path and model_error is None:
                    try:
                        if model is None:
                            os.environ["YOLO_OFFLINE"] = "true"
                            os.environ["YOLO_AUTOINSTALL"] = "false"
                            from ultralytics import YOLO
                            from ultralytics.utils.events import events
                            events.enabled = False
                            model = YOLO(str(self.model_path))
                        result = model.predict(
                            frame, device=self.device, imgsz=640,
                            conf=min(self.thresholds.values()), verbose=False,
                        )[0]
                        boxes = result.boxes
                    except Exception as error:
                        model_error = str(error)
                for box in boxes:
                    label = result.names[int(box.cls.item())]
                    if label == "person":
                        label = "human"
                    elif label in COCO_ANIMALS:
                        label = "animal"
                    confidence = float(box.conf.item())
                    if label in self.thresholds and confidence >= self.thresholds[label]:
                        detections.append({"class": label, "confidence": confidence,
                                           "xyxy": box.xyxy[0].tolist()})
                detections = tracker.update(detections)
                for box in detections:
                    x1, y1, x2, y2 = map(int, box["xyxy"])
                    color = (255, 220, 90) if box["class"] == "human" else (130, 240, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{box["class"]} #{box["track_id"]} {box["confidence"]:.0%}',
                                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("Could not encode camera frame.")
                frames += 1
                with self.lock:
                    if self.stop_event.is_set():
                        break
                    self.jpeg = encoded.tobytes()
                    self.state = {"status": "live", "detections": detections,
                                  "frames": frames, "updated": time.time()}
                    if model_error is not None:
                        self.state["detection_error"] = model_error
        except Exception as error:
            with self.lock:
                if not self.stop_event.is_set():
                    self.jpeg = None
                    self.state = {"status": "error", "error": str(error), "detections": [], "frames": 0}
        finally:
            if capture is not None:
                capture.release()
