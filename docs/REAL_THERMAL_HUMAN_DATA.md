# Real thermal human data pilot

HIT-UAV was selected as supplemental real aerial thermal person data. Its
official repository describes images captured at heights of 60 to 130 meters,
camera angles from 30 to 90 degrees, and both day and night conditions.
The source is https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
and the paper is https://doi.org/10.1038/s41597-023-02066-6.
The repository distributes the data under CC-BY-4.0.

`scripts/prepare_real_thermal_humans.py` imported 450 human-containing
images from the source training export. It uses its standard bounding boxes
and maps source Person class 0 to Reserve Watch human class 1. Source
vehicles are outside the animal/human task and their labels are omitted.
Every downloaded image was decoded and checked against its annotated
dimensions. Human boxes were checked for positive size and image bounds.
The source files, URLs, hashes, class mapping and selection are retained under
`data/external/hit_uav_humans_v1/`.

The 350 training images contain 2,079 human boxes from seven capture dates.
The 100 external validation images contain 627 human boxes from dates
20210114 and 20210118, which are entirely excluded from training. Date and
filename metadata groups were used to spread the selected frames. The
public metadata did not establish recording IDs, so date separation is
the conservative proxy used here. Cross-split exact image hashes were
checked. Similar settings and locations can still appear on different dates.

`scripts/prepare_hit_uav_pilot.py` prepares a 1,500-frame mixed pilot with
1,000 BIRDSAI animal-only frames, 150 BIRDSAI human-containing frames spread
across the four real human sequences, and the 350 new HIT-UAV training
frames. It retains the original 3,098-frame real BIRDSAI validation set
and checks training-image hashes against that set, TestReal and external
validation. The two validation sets measure different scene distributions.
HIT-UAV includes urban scenes, so it supplements the park data with
real people under different camera conditions.

Before additional training, original epoch 9 matched 3 of 627 external
human boxes at confidence 0.25 and matching IoU 0.50, with 17 false
human detections. At confidence 0.10 it matched 10 with 59 false
detections. Evidence is in
`outputs/evaluation/hit_uav_20261117_222105_680793.json`.
The reusable evaluation entry point is
`scripts/evaluate_external_humans.py --model CHECKPOINT`.

## Completed three-epoch pilot

The mixed dataset contains 2,316 human boxes and 3,433 animal boxes across
1,500 training frames. Training started from original epoch 9 with
AdamW at 0.00005, no warmup, and the first 22 modules frozen. It ran
three epochs on Intel Arc. The saved run is
`runs/birdsai_low_lr_freeze22_20261117_222406_925673`.

At confidence 0.25 and matching IoU 0.50 on held-out HIT-UAV dates,
correct human boxes increased from 3 to 46 of 627, with false human
detections increasing from 17 to 112. On original real BIRDSAI validation,
the pilot matched 50 of 2,697 humans, including 37 of 2,618 small
human boxes, with 818 false human detections. The original checkpoint
matched 11 humans, all small, with 260 false detections using the same
fixed-threshold evaluator. Both human recall and precision increased on
each validation dataset, alongside a higher number of false detections.

The real-only frozen-22 control matched 15 humans with 445 false
detections on BIRDSAI validation. The comparison changes 350
human-containing frames from BIRDSAI to HIT-UAV, so it tests both
source variety and increased human exposure. These results are from one
seed and one short training budget. TestReal and field-camera performance
of the mixed-data checkpoint remain unverified.

The BIRDSAI fixed results are saved in
`outputs/evaluation/human_pilot_fixed_20261117_223144_045157/fixed_results.json`.
The external results are saved in the latest `outputs/evaluation/hit_uav_*.json`
with the checkpoint hash
`dbd2de1a570d2b62ca59ef2e959081798dbfdfd366617c0d378b04569fc33bc0`.

## Pending full TestReal comparison

A paired evaluation of original epoch 9 and the mixed-data checkpoint
was attempted with identical batch 8 settings and CPU NMS. Automatic
approval review rejected continuation after reporting an unexpected
external Visual Studio telemetry connection with unknown payload. No
completed new TestReal result was present when the local output folders
were checked. The existing validated checkpoint files remain available.
The mixed-data TestReal comparison is still pending.

The user explicitly approved the reported telemetry connection. A retry
was rejected by automatic review because the payload remained unverified.
Read-only inspection confirmed that the project runtime disables
Ultralytics events and automatic installation. The import startup hooks
inspected contained only the setuptools distutils shim. These checks
do not identify or establish the payload of the reported Visual Studio
connection, so they do not resolve the execution block.

## Comparison runner

`scripts/compare_checkpoints.py` supplies both checkpoint paths, identical
TestReal settings, and the configured Intel Arc interpreter. It sets optional
telemetry controls before importing inference libraries and installs a Python
audit hook that rejects socket connections, DNS resolution and datagram
sends. The `--check-network-guard` self-check tests that the connection
event raises before a socket connection is made.

The audit hook covers Python events. Native library networking and activity
from the execution host are outside that boundary. Consequently, this code
change does not establish that the previously reported connection has been
disabled, and it does not resolve the automatic review rejection by itself.
Only argument parsing and the network-guard self-check were run during
preparation. Full model evaluation remains pending verification of the
reported external activity.

`scripts/run_offline_comparison.ps1` adds an OS-level mitigation for native
network traffic originating from the configured Python interpreter. It
requires an elevated PowerShell, creates and verifies a temporary outbound
block rule, invokes the comparison, and removes the rule in a finally block.
Its PowerShell syntax was checked. Applying the firewall rule and running
the comparison have not been tested because the current session lacks
administrator rights. Traffic originating from a different host process
is outside the rule's scope, so this still does not establish the source
of the earlier reported connection.
