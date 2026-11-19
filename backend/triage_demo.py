"""Run the frontend and local triage API from the wildlife repository."""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageDraw, ImageFont

from src.tracking import DetectionTracker
from backend.live_video import LiveCameras

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT / "outputs/predictions/prediction_20261111_211648_193762"
DEFAULT_STATE = PROJECT / "outputs/triage_demo/reviews.json"
AREA = "North Trail"
EXPECTED_HEADCOUNT = 1
OUTCOMES = ("Confirmed ranger", "Other authorised person", "Unverified")
PATROLS = [
    {"team": "Patrol Alpha", "area": AREA, "departure": "18:00", "returnTime": "18:45"},
    {"team": "Patrol Bravo", "area": "River Gate", "departure": "18:10", "returnTime": "19:00"},
]
FRONTEND_ORIGINS = {"http://127.0.0.1:5173", "http://localhost:5173"}


def draw_tracks(image, detections, show_ids=True):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(10, min(16, image.width // 30)))
    for box in detections:
        x1, y1, x2, y2 = map(round, box["xyxy"])
        color = "#62dfff" if box["class"] == "human" else "#b5f285"
        label = box["class"].title()
        if show_ids:
            label += f' #{box["track_id"]}'
        corners = max(2, min(10, (x2 - x1) // 3, (y2 - y1) // 3))
        for x, dx in ((x1, 1), (x2, -1)):
            for y, dy in ((y1, 1), (y2, -1)):
                points = [(x + dx * corners, y), (x, y), (x, y + dy * corners)]
                draw.line(points, fill="#101714", width=4)
                draw.line(points, fill=color, width=2)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = right - left + 6, bottom - top + 4
        x = max(0, min(x1, image.width - label_width))
        y = max(0, y1 - label_height)
        draw.rectangle((x, y, x + label_width, y + label_height), fill="#101714")
        draw.text((x + 3, y + 2 - top), label, fill=color, font=font)


def load_frames(run, start, count):
    frames, images, episodes = [], [], {}
    start_time = datetime(2026, 11, 12, 18, 20, tzinfo=timezone(timedelta(hours=3)))
    last_excess = None
    tracker = DetectionTracker()
    with (run / "detections.jsonl").open(encoding="utf-8") as records:
        for source_index, line in enumerate(records):
            if len(frames) == count:
                break
            record = json.loads(line)
            detections = tracker.update(record["detections"])
            if source_index < start:
                continue
            image = run / Path(record["source"]).name
            source = Path(record["source"])
            render_tracks = source.is_file()
            if not image.is_file() and not render_tracks:
                raise FileNotFoundError(f"Annotated frame missing: {image}")
            index = len(frames)
            time = (start_time + timedelta(seconds=source_index)).isoformat()
            humans = [box for box in detections if box["class"] == "human"]
            excess = max(0, len(humans) - EXPECTED_HEADCOUNT)
            if excess:
                if last_excess is None or index - last_excess > 8:
                    episode = f"headcount/{run.name}/{image.name}"
                last_excess = index
                episodes[index] = {
                    "id": episode, "name": image.name, "time": time, "frame": index,
                    "count": len(humans), "confidence": max(box["confidence"] for box in humans),
                    "expected": EXPECTED_HEADCOUNT, "excess": excess,
                    "track_ids": [box["track_id"] for box in humans],
                    "human_detections": [
                        {"track_id": box["track_id"], "confidence": box["confidence"]}
                        for box in humans
                    ],
                    "camera": "CAM-01", "area": AREA,
                }
            images.append(source if render_tracks else image)
            frames.append({"name": image.name, "image": f"/api/frames/{index}",
                           "time": time, "detections": detections,
                           "expected": EXPECTED_HEADCOUNT, "excess": excess,
                           "render_tracks": render_tracks})
    if not frames:
        raise ValueError("The selected run and frame range contain no frames")
    return frames, images, episodes


def matching_patrols(alert, patrols):
    at = alert["time"][11:16]
    teams = []
    for patrol in patrols:
        departure, expected = patrol["departure"], patrol["returnTime"]
        on_duty = departure <= at <= expected if departure < expected else at >= departure or at <= expected
        if patrol["area"].strip().casefold() == alert["area"].strip().casefold() and on_duty:
            teams.append(patrol["team"])
    return teams


class ReviewStore:
    def __init__(self, path, episodes):
        self.path = path
        self.episodes = episodes
        self.lock = Lock()

    def request(self, action, data):
        with self.lock:
            state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
                "patrols": PATROLS.copy(), "alerts": [],
            }
            patrols, alerts = state["patrols"], state["alerts"]
            changed = False
            if action == "observe":
                index = data.get("frame")
                if type(index) is not int:
                    raise ValueError("A frame number is required.")
                alert = self.episodes.get(index)
                if alert and not any(
                    a["id"] == alert["id"] or (a.get("excess") and not a.get("outcome"))
                    for a in alerts
                ):
                    alerts.append(alert.copy())
                    changed = True
            elif action == "patrols":
                team, area, departure, expected = (data.get(k) for k in ("team", "area", "departure", "returnTime"))
                if not isinstance(team, str) or not team.strip() or len(team) > 60:
                    raise ValueError("Enter a team name up to 60 characters.")
                if not isinstance(area, str) or not area.strip() or len(area) > 60:
                    raise ValueError("Enter a patrol location of up to 60 characters.")
                area = area.strip()
                if departure == expected or not all(
                    isinstance(value, str) and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value)
                    for value in (departure, expected)
                ):
                    raise ValueError("Choose an area and distinct valid patrol times.")
                patrols.append({"team": team.strip(), "area": area,
                                "departure": departure, "returnTime": expected})
                changed = True
            elif action == "reviews":
                outcome, note = data.get("outcome"), data.get("note", "")
                if outcome not in OUTCOMES or not isinstance(note, str) or len(note) > 500:
                    raise ValueError("Choose a review outcome and enter up to 500 characters.")
                alert = next((a for a in alerts if a["id"] == data.get("id") and not a.get("outcome")), None)
                if alert is None:
                    raise ValueError("This alert is no longer open.")
                alert.update(outcome=outcome, note=note.strip(), reviewed_at=datetime.now(timezone.utc).isoformat())
                changed = True
            elif action == "reset":
                alerts.clear()
                changed = True
            if changed:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                pending = self.path.with_suffix(".tmp")
                pending.write_text(json.dumps(state, indent=2), encoding="utf-8")
                pending.replace(self.path)
            return {"patrols": patrols, "alerts": [
                {**alert, "matching_teams": matching_patrols(alert, patrols)} for alert in alerts
            ]}


def make_handler(frames, images, store, port, legacy_ui=False, live=None):
    page = (PROJECT / "demos/coursework/index.html").read_bytes() if legacy_ui else None
    image_routes = {frame["image"]: (image, frame) for frame, image in zip(frames, images)}
    animal_video = "birdsai-0357-animal-detections.webm"
    if not (PROJECT / "outputs" / animal_video).is_file():
        animal_video = "birdsai-0357-preview.webm"
    video_sources = (
        (animal_video, "BIRDSAI 0357 · animal detection test", "video/webm"),
        ("nz-thermal-1486055-detections.webm", "Thermal recording · clip 1486055", "video/webm"),
        ("nz-thermal-720986-detections.webm", "Thermal recording · deer", "video/webm"),
        ("nz-thermal-581381-detections.webm", "Thermal recording · wallaby", "video/webm"),
        ("nz-thermal-1480305-detections.webm", "Thermal recording · kiwi", "video/webm"),
        ("bu-tiv-detection-preview.webm", "BU-TIV atrium · model predictions", "video/webm"),
        ("bu-tiv-atrium-orange.mp4", "BU-TIV atrium · original footage", "video/mp4"),
    )
    videos = {}
    for filename, label, mime in video_sources:
        path = PROJECT / "outputs" / filename
        stable = path.with_stem(path.stem + "-tracks")
        if stable.is_file() and stable.stat().st_size:
            path = stable
        if path.is_file() and path.stat().st_size:
            videos[f"/api/videos/{filename}"] = (path, label, mime)

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, body, mime, status=200):
            self.send_response(status)
            self.send_origin_headers()
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_origin_headers(self):
            origin = self.headers.get("Origin")
            if origin in FRONTEND_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def do_OPTIONS(self):
            if self.headers.get("Origin") not in FRONTEND_ORIGINS:
                return self.send_json({"error": "Frontend origin is not allowed."}, 403)
            self.send_response(204)
            self.send_origin_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def send_json(self, payload, status=200):
            self.send_body(json.dumps(payload).encode(), "application/json", status)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/live":
                self.send_body((PROJECT / "backend/live.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/dashboard":
                frontend = PROJECT.parent / "wildlife-frontend/index.html"
                if not frontend.is_file():
                    return self.send_body(
                        b'<p>Place the wildlife-frontend repository beside the backend.</p><a href="/live">Camera setup</a>',
                        "text/html; charset=utf-8", 404,
                    )
                dashboard = frontend.read_text(encoding="utf-8-sig")
                dashboard = dashboard.replace(
                    '<script src="config.js"></script>',
                    '<script>window.reserveWatchApi = location.origin;</script>',
                    1,
                )
                panel = (PROJECT / "backend/live_panel.html").read_text(encoding="utf-8")
                layout = (PROJECT / "backend/camera_wall.css").read_text(encoding="utf-8")
                dashboard = dashboard.replace("</style>", f"\n{layout}\n</style>", 1)
                dashboard = dashboard.replace("</main>", panel + "</main>", 1)
                self.send_body(dashboard.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/api/live/status" and live is not None:
                self.send_json(live.snapshots())
            elif route == "/api/live/frame" and live is not None:
                camera_id = parse_qs(urlsplit(self.path).query).get("id", [""])[0]
                state, jpeg = live.frame(camera_id)
                if jpeg and state["status"] == "live":
                    self.send_body(jpeg, "image/jpeg")
                else:
                    self.send_json({"error": "No live camera frame available."}, 503)
            elif route == "/":
                if page is None:
                    self.send_body(b'<h1>Reserve Watch</h1><p><a href="/live">Open live phone camera</a></p><p><a href="/dashboard">Open surveillance dashboard</a></p>', "text/html; charset=utf-8")
                else:
                    self.send_body(page, "text/html; charset=utf-8")
            elif route == "/api/frames":
                self.send_json(frames)
            elif route == "/api/state":
                self.send_json(store.request(None, {}))
            elif route == "/api/videos":
                self.send_json([{"url": url, "label": label} for url, (_, label, _) in videos.items()])
            elif route in videos:
                path, _, mime = videos[route]
                self.send_body(path.read_bytes(), mime)
            elif route in image_routes:
                path, frame = image_routes[route]
                if not frame["render_tracks"]:
                    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                    self.send_body(path.read_bytes(), mime)
                else:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                    draw_tracks(image, frame["detections"], show_ids=False)
                    data = BytesIO()
                    image.save(data, "JPEG", quality=90)
                    self.send_body(data.getvalue(), "image/jpeg")
            else:
                self.send_json({"error": "Route not found."}, 404)

        def do_POST(self):
            route = urlsplit(self.path).path
            action = route.removeprefix("/api/")
            if route != f"/api/{action}" or action not in ("observe", "patrols", "reviews", "reset", "live/connect", "live/disconnect"):
                return self.send_json({"error": "Route not found."}, 404)
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts or self.headers.get("Origin", "") not in (
                "", *(f"http://{host}" for host in hosts), *FRONTEND_ORIGINS
            ):
                return self.send_json({"error": "Open the dashboard from its local address."}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get_content_type() != "application/json" or not 0 < length <= 8192:
                    raise ValueError("Send a small JSON object.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Send a JSON object.")
                if action.startswith("live/"):
                    if live is None:
                        raise ValueError("Start the backend with --live-model pointing to a checkpoint.")
                    if action == "live/connect":
                        camera_id = live.connect(data.get("source"), data.get("name"), data.get("location", ""))
                        return self.send_json({"id": camera_id})
                    else:
                        live.disconnect(data.get("id"))
                    self.send_json({"status": "disconnected"})
                else:
                    self.send_json(store.request(action, data))
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, *_args):
            pass

    return Handler


def main(legacy_ui=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--live-model", type=Path, help="Animal/human checkpoint for live phone inference")
    parser.add_argument("--live-device", default="cpu", help="cpu or Intel Arc xpu:0")
    parser.add_argument("--live-animal-conf", type=float, default=0.25, help="Minimum score for live animal boxes")
    parser.add_argument("--live-human-conf", type=float, default=0.25, help="Minimum score for live human boxes")
    args = parser.parse_args()
    if args.start < 0 or args.frames < 1 or not 1 <= args.port <= 65535:
        parser.error("Choose a nonnegative start, at least one frame, and a valid port.")
    frames, images, episodes = load_frames(args.run.resolve(), args.start, args.frames)
    store = ReviewStore(args.state.resolve(), episodes)
    if args.live_model and not args.live_model.is_file():
        parser.error("The live model checkpoint does not exist.")
    thresholds = {"animal": args.live_animal_conf, "human": args.live_human_conf}
    if not all(0 < confidence <= 1 for confidence in thresholds.values()):
        parser.error("Live confidence thresholds must be greater than 0 and at most 1.")
    live = LiveCameras(args.live_model.resolve() if args.live_model else None, args.live_device, thresholds)
    handler = make_handler(frames, images, store, args.port, legacy_ui=legacy_ui, live=live)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Open http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if live:
                live.stop()


if __name__ == "__main__":
    main()
