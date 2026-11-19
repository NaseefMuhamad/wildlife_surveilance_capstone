# Live phone camera refinement — 18 November 2026

Added an opt-in live page to the existing backend at `/live`. Existing recorded results, checkpoints and review records are preserved. The backend remains bound to localhost. The phone publishes a camera stream on the LAN and the PC connects to it, so the phone does not need access to the backend.

## Run

From the backend repository, using an environment with working PyTorch, Ultralytics, Pillow and OpenCV:

```powershell
python -m backend.triage_demo --live-model runs/birdsai_corrected_20261114_234448_294790/weights/best.pt --live-device cpu
```

Open http://127.0.0.1:8765/live on the PC. Connect phone and PC to the same Wi-Fi. Start a phone camera app that exposes an HTTP/MJPEG feed. Paste the actual video endpoint supplied by that app, for example `http://192.168.1.50:8080/video`, then click Connect camera. A camera app's settings webpage is not a video endpoint. Guest networks with client isolation can prevent this connection. No router port forwarding is needed.

For Intel Arc, use the existing tested XPU Python environment and pass `--live-device xpu:0`. The server does not automatically select an interpreter. Do not use a broken Torch installation. Model loading errors appear in the live page.

## Processing and scope

OpenCV reads the stream on a background thread with five-second open/read timeouts. The selected checkpoint predicts at image size 640 with confidence threshold 0.25. Only animal/human labels are displayed. Existing class-aware IoU tracking assigns IDs. Bounding boxes and confidence values describe the processed frame. Browser polling requests the newest completed JPEG and status every 700 ms. No recording is saved. Stream buffering and inference can introduce latency, so this is a first single-phone integration, not a ten-camera production scheduler.

Disconnect clears old frames and detections. Errors and stalled feeds are explicitly displayed. A reconnect may require waiting for a pending camera read to finish. Live counts are frame counts, not counts of unique visitors. Live detections are not yet connected to patrol discrepancies or persisted review alerts. Class confidence still updates per frame. The separate frontend repository is unchanged. Access this first integration through the backend live-page URL.

Thermal BIRDSAI training differs from ordinary colour phone images. Successful video transport does not establish human detection accuracy. Test actual phone footage before claiming improved recall. No checkpoint is promoted by this feature.

## Validation

Unit checks cover URL validation, clearing stale detections on disconnect, and identifying stalled frames. A real phone Wi-Fi connection and inference speed require a phone stream provided by the operator and remain unverified.

For ordinary colour phone footage, pass --live-model yolov8n.pt. The live feed maps COCO person to human and supported animal species to animal. The thermal checkpoint remains available for thermal streams.
After you connect, the first processed frame opens /dashboard on the backend port. This route serves the adjacent wildlife-frontend dashboard with a live camera panel, so port 5173 is optional for this flow.
The first raw camera frame opens the dashboard before inference starts. Omit --live-model for video only. If optional inference fails, the camera continues in video-only mode.
Use Add live camera on the dashboard to register each phone with its own name and HTTP/MJPEG address. Up to eight live sources can be registered during a backend session. Disconnect operates on one tile. Start in video-only mode when checking concurrent streams.
