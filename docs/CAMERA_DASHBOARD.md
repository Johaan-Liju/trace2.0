# TRACE camera dashboard and monitoring

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
  clip_service.py           Collect entry clips and return finished MP4 paths
  utils.py                  Shared settings, drawing, geometry, and zone checks
config/
  camera.example.json       Example settings
  camera.local.json         Your settings (ignored by Git)
tests/                      Automated checks
docs/BUILD_GUIDE.md          Code walkthrough and current scope
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
# In process_frame(), alert mode follows this sequence:
detections = detect_and_track(frame, detector, tracker, timestamp)
detections = Utils.assign_zones(detections, zones, frame.shape)
entries = entry_alerts.check_entries(detections, tracker.get_remembered_ids())
entry_alerts.notify(entries)
preview = Utils.draw_frame_results(frame, detections, zones, detection_enabled=True)
```

Functions have docstrings and related functions sit under labeled blocks.
See the [function walkthrough](BUILD_GUIDE.md#function-walkthrough) for what
each step takes in, returns, and remembers.

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

### Browser dashboard

Launch the local frontend from the project folder:

```powershell
.\.venv\Scripts\python.exe dashboard.py
```

Open **http://127.0.0.1:8765**. The dashboard uses `config/camera.local.json` if it
exists, otherwise the example config. No additional packages or frontend build
step are needed. Choose a different config or port with `--config` and `--port`.

- **Overview:** start/stop the camera, watch annotated frames, and see current
  people, processing FPS, entry counts, and recent alerts.
- **Zones:** choose **Camera preview**, start and stop the camera to capture a
  frame, then draw a restricted or ignore polygon with at least three corners.
  You can edit or remove saved zones here. Stop monitoring before making changes.
- **Activity:** view the most recent 500 restricted-entry alerts. The alert total
  includes all entries since the dashboard server started, across camera runs.
- **Recordings:** download completed entry clips from this server session.
  Enable **Record entry clips** in Settings; FFmpeg is required. Files remain on
  disk in the configured output folder after the server exits.
- **Settings:** save the source, camera name, new-person confidence, alert sound,
  zone-alert toggle, and clip toggle. Confidence updates detection and ByteTrack's
  high/new thresholds together; it must exceed the configured low threshold.

**AI monitoring** runs the existing YOLO + ByteTrack pipeline. Restricted-entry
alerts require a restricted zone and the alerts toggle. Clips respond to people
appearing anywhere in view. **Camera preview** displays video without inference,
alerts, or recording. A stopped/ended feed is explicitly labeled as the last frame.

The server listens only on this computer's loopback interface. Keep it running
while using the browser; Ctrl+C stops the server and releases the camera. Stop
the command-line monitor or zone editor before using the same camera here.
Model startup, camera retries, and finishing a clip may take time; Stop waits for
in-flight operations to finish. Browser refreshes do not stop monitoring.

Settings are saved to the selected JSON file; private `TRACE_CAMERA_SOURCE`
overrides remain in the environment. RTSP source credentials are not returned to
the browser. Alert history and the downloadable-clip list are held in memory for
the server session, not persisted as an incident database. This is a single-camera
local dashboard, not an authenticated remote hosting service.

### Command-line monitor

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

## Return clips when someone appears

```powershell
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --clips
```

`--clips` enables person detection and tracking automatically. A person appearing
anywhere in view starts a five-second clip. No zone is needed. Someone already
visible at startup also counts. The console prints `CLIP_READY: <absolute path>`
after FFmpeg finishes a playable MP4 in `storage/clips`. Keep or move that file
wherever you want; this feature does not upload it.

Add this optional block to your camera JSON to change the defaults:

```json
"clips": {
  "output_dir": "storage/clips",
  "duration_seconds": 5,
  "fps": 15,
  "reentry_gap_seconds": 2
}
```

- No detected person means no clip. Remaining visible does not start repeated clips.
- A return after at least two seconds without a detection counts as another entry,
  even if tracking keeps the same ID. Shorter gaps are treated as missed detections.
- People arriving during an active clip share that clip; its end time stays fixed.
- Stopping, file EOF, or reconnection finishes a shorter clip with frames collected so far.
- Clips contain camera images from the first detection onward, without audio or
  preview labels. There is no footage from before the entry. Ignore zones apply to
  restricted-area alerts; they do not suppress these whole-frame clips.
- Clip FPS controls sampling/output rate. Live timing uses elapsed seconds; files
  use their source timeline. Slow detection means fewer distinct frames, so the
  encoder holds the available images to preserve elapsed time. Missed people cannot
  trigger clips, and tracking ID changes or longer detection gaps can trigger extras.
- FFmpeg must be on `PATH` (`ffmpeg -version` checks it). Install it separately if it is missing from your machine. Encoding briefly pauses processing when a clip finishes.

The small `ClipRecorder` class keeps entry history and temporary frames.
`update()` returns a finished `Path` or `None`; `close()` returns the last shorter
clip when stopping. Both print completed paths. `Utils.encode_clip()` handles the
FFmpeg command and deletes unfinished output on failure. Temporary JPEGs are
cleaned after encoding. Completed MP4s remain until you move or delete them.

You can combine `--clips --alerts` to keep restricted-zone sound/banner alerts
alongside whole-frame entry clips. Encoding uses FFmpeg's
[concat timing and MP4 output options](https://ffmpeg.org/ffmpeg-formats.html).

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

The camera checks cover playback, cleanup, configuration, actual ByteTrack matching,
zone geometry, editor saving/canceling, and immediate entry alerts. Alert checks
cover repeat suppression, exits/re-entry, missed detections, and ignore zones. They require no
camera or model download. Sample inference and video checks exercise real YOLO.
Clip checks cover empty views, continuous presence, re-entry, shared clips,
reconnection, failure cleanup, and a real playable MP4 with elapsed-time checks.
The real encoder test is skipped if FFmpeg is unavailable.
Dashboard checks exercise actual HTTP requests, source privacy, polygon validation,
start/stop cleanup, duplicate-start prevention, previews, and clip downloads.
Your actual camera and desktop editor still need manual checks on your machine.

The immediate-alert video check produced exactly two person-entry alerts across
five frames, without repeats while they remained inside. Sound was muted for that
automated run; confirm the system sound on your computer during the manual check.

The focused project scope and code flow are in `docs/BUILD_GUIDE.md`.
