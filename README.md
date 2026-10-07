# TRACE

This project combines the camera dashboard from `Johaan-Liju/trace2.0` with the existing restricted-zone prototype and trained violence-video classifier.

## Setup

From PowerShell in the repository root, install Python 3.12 and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
if (!(Test-Path config/camera.local.json)) {
    Copy-Item config/camera.example.json config/camera.local.json
}
```

Keep an existing virtual environment if it already works. `requirements-tested.txt` records the environment verified with both pipelines; `requirements.txt` contains the minimal direct dependencies. FFmpeg must be on `PATH` to record entry clips. The root dashboard requires no Node.js or frontend build.

## Camera dashboard

```powershell
.\.venv\Scripts\python.exe dashboard.py
```

Open **http://127.0.0.1:8765**. In Settings, set the source to `0` for a webcam or a video file path; the example's `storage/demo.avi` is a placeholder. Use Camera preview to capture a frame, stop the camera, then draw a restricted zone in Zones. Start AI monitoring to detect people, track temporary IDs, and alert immediately on entry. Recordings can save clips when people appear.

See the [camera dashboard guide](docs/CAMERA_DASHBOARD.md) for settings, the zone editor, CLI modes, clip recording, and limits.

**Violence review** connects the trained classifier to the same dashboard. Upload
a video, select **Analyse violence** on a recording, or enable **Record entry
clips** and **Automatically analyse entry clips** in Settings. Completed clips
are queued for background analysis. The review page shows progress, cancellation,
video playback, a score timeline, candidate timestamps, and a downloadable JSON
report. See the [integrated workflow](docs/VIOLENCE_DASHBOARD.md).

Automatic analysis checks recorded entry clips after they finish; it is not
continuous violence detection across the live feed. Restricted-zone entry alerts
remain separate from experimental violence-review results.

## Trained violence classifier

Double-click `run_violence.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\path\to\short-video.mp4" --scan
```

This CPU-capable model scores a complete video as possible violence or no violence flag. It does not identify stalking or locate the exact event time. It needs both `models/r3d_18-b3b3357e.pth` and `runs/violence_baseline/best.pt`. These files are present in this local Git history; if they are absent in another checkout, use the model ZIP described in the [clone setup guide](docs/FRIEND_SETUP.md).

The launcher now scans overlapping windows throughout the video and reports
candidate review timestamps. This addresses gaps between the original three
samples. Scan scores and thresholds are experimental and can produce more false
alarms; the existing dataset accuracy does not apply to this mode. Omit `--scan`
to compare with the original whole-video prediction. See the
[scan guide](docs/VIOLENCE_TRAINING.md#investigate-missed-events-with-a-window-scan).

- [Training, including another dataset](docs/VIOLENCE_TRAINING.md)
- [Measured results and evaluation limits](docs/VIOLENCE_RESULTS.md)
- [Dataset options](docs/DATASETS.md)

## Existing zone prototype and person training

`run.cmd` and `python -m trace.monitor` retain the original dwell-based zone monitor, snapshots, JSONL records, and SQLite events. Its `zone.local.json` format is separate from the dashboard's `config/camera.local.json`; configure zones for the interface you use. Close one monitor before using the same camera in another.

The [original prototype guide](docs/PROTOTYPE_GUIDE.md) covers frame extraction, person annotation, dataset validation, and YOLO training. The [training guide](docs/TRAINING.md) adds evaluation guidance.

## Project layout

| Path | Purpose |
| --- | --- |
| `dashboard.py`, `frontend/`, `services/` | Browser dashboard and camera services |
| `main.py`, `edit_zones.py`, `config/` | Camera CLI and zone editor |
| `trace/` | Existing monitor, video classifier, and training |
| `scripts/` | Dataset preparation and model packaging |
| `tests/` | Checks for both TRACE pipelines |
| `clipcraft/` | Separate reference application imported with the source repository; optional and not used by TRACE |

## Verify

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) '.runtime\ultralytics'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The combined suite includes 76 tests covering tracking, zones, alerts, HTTP dashboard requests, video decoding, clip encoding, dataset import, and classifier utilities. Real clip encoding requires FFmpeg. Live camera behavior and alert sound still need checks on the target laptop.
