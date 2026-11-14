# Reserve Watch roadmap

## Backend and project evidence

| Path | Contents |
|---|---|
| `backend/` | Replay API, alerts, patrol register, and review storage |
| `src/` | Dataset preparation, training, inference, evaluation, and tracking |
| `configs/` | Dataset and training configuration |
| `tests/` | Checks for dataset preparation and evaluation |
| `demos/coursework/` | Original dashboard retained for the coursework demo |
| `scripts/diagnostics/` | Dataset inspection scripts and early inference experiments, run from the repository root |
| `docs/` | Progress report, corrections, original label audit, and verification log |
| `data/`, `models/`, `weights/` | Local datasets and model files |
| `runs/detect/runs/birdsai_yolov8n-2/` | Original training curves, metrics, and `weights/best.pt` |
| `outputs/` | Prediction runs, replay videos, and saved ranger reviews |

Datasets, checkpoints, and run outputs are local artifacts. Keep these available
when demonstrating the historical results. A fresh clone needs its own copies
of the demo media and saved predictions.

### Where to find the results

| Evidence | Location |
|---|---|
| Training metrics and charts | `runs/detect/runs/birdsai_yolov8n-2/results.csv` and `results.png` in the same run directory |
| Precision-recall curve and confusion matrix | `BoxPR_curve.png` and `confusion_matrix.png` in the same run directory |
| North Trail replay predictions | `outputs/predictions/prediction_20261111_211648_193762/` |
| Demo review history | `outputs/triage_demo/reviews.json` |
| Coursework milestone report | [Group 5 Final CV DL Milestone Report](https://chatgpt.com/library?file_id=libfile_8fd9d4dd490881919017fcb034188af4&account_id=personal) |

## Delivery order

### Dashboard layout refinement

The standalone frontend applies the same 4:3 viewport to every Camera wall
feed. Each image or clip is fitted inside its viewport so the full frame and
its detection boxes remain visible. The grid reduces its column count on
narrow screens. Focus and review modes give North Trail a wider viewport
for inspecting detections. This refinement changes the dashboard presentation.

See [BIRDSAI dataset split](DATASET_SPLIT.md) for the rebuild method,
class counts, and audit evidence for the next training run.

| Stage | Work | Verification |
|---|---|---|
| Repository separation | Run the standalone frontend against the Python API. Keep the original demo entry point available. | North Trail opens the two-detected, one-expected alert and a ranger can save a review. |
| Recorded camera wall | Catalog roughly ten clips with camera IDs and timestamped detections. Give each feed playback controls, detections, counts, and an alert link to its source frame. | Play the wall on the demo PC and verify synchronized overlays and correct camera attribution. |
| Live Wi-Fi camera | Confirm the camera's local stream format, add ingestion, and show connection status. Run detection and tracking on sampled frames. | Demonstrate the local stream and measure preview and alert latency. Evaluate RGB or thermal footage according to the actual camera. |
| Model evaluation | Verify the disjoint sequence split and class coverage, train a fresh checkpoint, and compare it with the original on the same held-out clips. | Report human and animal scores, missed alerts, false alerts, and ID switches. |

## Local setup

Start the API from the backend repository with
`.\.venv\Scripts\python.exe -m backend.triage_demo` on port 8765.
The standalone frontend repository is at `C:/workspace/reserve-watch-frontend`
on this PC. Run `python -m http.server 5173 --bind 127.0.0.1` from that
directory and open http://127.0.0.1:5173.
