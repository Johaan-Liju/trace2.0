# TRACE

A security camera that alerts immediately when a person enters a restricted area.
Use YOLO, ByteTrack, and your drawn zones. No dwell timer or loitering rule is needed.

## Simple code structure

The structure follows the pattern in the local Clipcraft project: short entry
scripts, feature service files, and shared functions in one `Utils` class.

```text
main.py                     Start monitoring
edit_zones.py               Start the zone editor
services/
  video_service.py          Video loop, camera, YOLO, and ByteTrack
  zone_service.py           Mouse zone editor
  event_service.py          Immediate entry alerts
  utils.py                  Shared settings, drawing, geometry, and zone checks
config/
  camera.example.json       Example settings
  camera.local.json         Your settings (ignored by Git)
tests/                      Automated checks
docs/BUILD_GUIDE.md          Build order and future work
storage/                    Optional snapshots and local model settings
clipcraft/                  Reference project
```

Read the code in this order:

1. `main.py` reads settings and calls `run_video()`.
2. `services/video_service.py` starts with `run_video()` and `process_frame()`.
   Those functions show the main flow. Below them, labeled sections contain the
   camera, model, and tracker classes that retain state between frames.
3. `services/utils.py` holds the reusable `Utils` functions.
4. `services/zone_service.py` contains the editor's mouse, drawing, and save methods.
5. `services/event_service.py` compares previous and current zone membership and alerts on entry.

The frame-processing steps are direct function calls:

```python
# In process_frame(), tracking mode follows this sequence:
detections = detector.detect(frame, confidence=tracker.settings["track_low_thresh"])
tracks = tracker.update(detections, frame.shape, timestamp)
tracks = Utils.assign_zones(tracks, zones, frame.shape)
counts = Utils.count_zone_occupants(tracks, zones)
preview = Utils.draw_zones(frame, zones, counts)
preview = Utils.draw_detections(preview, tracks)
```

Functions have docstrings and related functions sit under labeled blocks.

## Setup

From the project folder in PowerShell, on a fresh installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config/camera.example.json config/camera.local.json
```

Keep your existing local config if you already completed setup. Set `source` to
`"0"` for a webcam, a video path such as `"C:/Videos/test.mp4"`, or an RTSP URL.
Use forward slashes in JSON file paths. Relative paths start at the working folder.

## Run the security camera

After saving a restricted zone, run:

```powershell
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --alerts
```

`--alerts` enables detection and tracking automatically. Each tracked person alerts
on their first observed entry into each restricted zone. Someone already inside
when monitoring starts also triggers an alert. There is no waiting period.

Alerts currently appear **on this computer**:

- The Windows alert sound plays asynchronously.
- The preview shows the most recent entry and its UTC time.
- The terminal prints every entry, including the camera, person ID, and zone.

Continuous presence does not repeat the alert. An observed exit followed by
re-entry does. A briefly missed detection is not treated as an exit; this avoids
repeating alerts when the person was merely occluded. If a person leaves and
returns entirely between observations, that unseen exit cannot be confirmed.
A new tracking ID may produce a fresh alert. Reconnecting the camera starts a
fresh observation. Ignore zones suppress restricted-area alerts.

To mute sound while keeping visual and terminal alerts, add this JSON field:

```json
"alerts": {"sound": false}
```

Sound is enabled by default on Windows. Phone/email delivery and persistent alert
history are not configured. The latest-entry banner remains visible as a record
of the last alert; it does not mean the person is still inside.

## Preview and diagnostic modes

```powershell
# Video only
.\.venv\Scripts\python.exe main.py --config config/camera.local.json

# Person detection
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --detect

# Detection, tracking IDs, and configured zones
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --track
```

Press Q in the preview, close its window, or press Ctrl+C to stop.
A local file stops at its end. A disconnected live camera retries according to
its settings. Failed startup connections require restarting the command.

For a bounded run without a preview window:

```powershell
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --track --headless --max-frames 100 --snapshot storage/preview.jpg
```

The snapshot is the first annotated frame. Recording and incident evidence are
not automatic yet. File EOF and some decoding failures share OpenCV's failed-read
signal; the program reports that ambiguity.

## Draw zones

Stop monitoring before opening the editor, so the webcam is available:

```powershell
.\.venv\Scripts\python.exe edit_zones.py --config config/camera.local.json
```

Click at least three corners in order, then click the green **Save** button.
Clicking the first corner again is optional. The editor uses a frozen camera
frame and places its buttons below the image.

| Action | Button or shortcut |
| --- | --- |
| Save | Save, S, or Enter |
| Undo a corner | Undo, Backspace, or right-click |
| Redraw | Clear or R |
| Cancel | Cancel, Q, Escape, or close window |

To add another zone, give it another ID. Reuse the ID to edit it:

```powershell
.\.venv\Scripts\python.exe edit_zones.py --config config/camera.local.json --zone-id ignored_01 --name "Ignore Area" --type ignore
```

Use `restricted` for alert areas and `ignore` for excluded areas. Remove a zone's object
from the JSON to delete it. Restart monitoring after changing settings.

Zone points are fractions from 0 to 1: `[0.5, 0.5]` means the image center. They
scale when the same camera view changes resolution. Redraw them if the view moves
or is cropped. Example field inside the camera config:

```json
"zones": [
  {
    "id": "restricted_01",
    "name": "Restricted Area",
    "type": "restricted",
    "points": [[0.1, 0.4], [0.6, 0.4], [0.6, 0.9], [0.1, 0.9]]
  }
]
```

Membership uses the bottom-center dot of each person's box. Boundaries count as
inside. Overlapping zones are supported; ignore zones override active memberships
and their counts. Ignored people remain tracked, with gray boxes. A restricted-zone
box indicates membership. Start with `--alerts` to enable entry notifications.

## Settings and limits

The example JSON shows all defaults. `Utils` validates settings before use.

- **Camera:** `display_width`, `fallback_fps`, `reconnect_attempts`,
  `reconnect_delay_seconds`, and RTSP `timeout_ms`.
- **Detection:** pretrained `yolov8n.pt`, confidence `0.4`, image size `640`,
  device `"cpu"`, and person class `[0]`. The first model download needs internet.
  A compatible CUDA installation can use device `"0"`; actual speed depends on
  hardware. No custom model training is required.
- **Tracking:** `track_buffer_seconds` defaults to five seconds of source frames.
  Your local config may override this, including the eight-second setting used
  for brief exits. The buffer converts using source FPS; slow live processing
  can make it last longer in wall-clock time. An explicit legacy `track_buffer`
  is supported; seconds take precedence when both are provided.
- **Tracking thresholds:** high `0.4`, low `0.1`, new track `0.4`, matching `0.8`,
  and score fusion enabled. Tracking passes low-confidence candidates to YOLO's
  output so ByteTrack can recover weak matches. Detection-only mode uses its own
  confidence threshold.

IDs are temporary and camera-local. Position and motion matching cannot guarantee
the same ID after someone leaves the view. Reconnection resets stale tracking
state; displayed IDs keep increasing within the run. No identity recognition is
implemented. Keep class `[0]` for the current person-tracking pipeline.

Detection dictionaries contain class name/ID, confidence, and box coordinates.
Tracks also contain `track_id`, `first_seen`, and `last_seen`. Times use estimated
frame-count/FPS video time for files and monotonic time for live cameras. Zone
functions add `zone_ids`, `zone_names`, `zone_types`, `current_zone`, and `ignored`.

For private RTSP sources, set `TRACE_CAMERA_SOURCE` in your shell instead of
saving credentials in JSON. Native OpenCV diagnostics may include source details;
check logs before sharing them. Editor saves preserve the raw configured source.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The 47 checks cover playback, cleanup, configuration, actual ByteTrack matching,
zone geometry, editor saving/canceling, and immediate entry alerts. Alert checks
cover repeat suppression, exits/re-entry, missed detections, and ignore zones. They require no
camera or model download. Sample inference and video checks exercise real YOLO.
Your actual camera and desktop editor still need manual checks on your machine.

The immediate-alert video check produced exactly two person-entry alerts across
five frames, without repeats while they remained inside. Sound was muted for that
automated run; confirm the system sound on your computer during the manual check.

The focused project scope and code flow are in `docs/BUILD_GUIDE.md`.
