# TRACE

Phases 1–4 of the supplied implementation plan: video input, YOLO person detection,
ByteTrack tracking, and polygon zones. Incidents, backend, and dashboard are later phases.

## Run it

Run these commands from this project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config/camera.example.json config/camera.local.json
```

Edit `source` in `config/camera.local.json`:

- Local video: `"storage/demo.avi"` or `"C:/Videos/test.mp4"`.
- USB webcam: `"0"` (a string containing the device number).
- CCTV: `"rtsp://camera-host:554/stream"`.

The example video is a placeholder: supply your own file. Relative video paths
are resolved from the directory where you run the command.

```powershell
.\.venv\Scripts\python.exe main.py --config config/camera.local.json
```

Press **Q** in the preview, close the preview window, or press **Ctrl+C** in the
terminal. A local video stops at its end. A dropped live stream retries three
times by default, then exits with an error. Restart the program if the source
is unavailable at startup. Configuration changes take effect on restart.

To provide a private URL without writing it into the JSON file:

```powershell
$env:TRACE_CAMERA_SOURCE = 'rtsp://user:password@camera-host:554/stream'
.\.venv\Scripts\python.exe main.py --config config/camera.local.json
Remove-Item Env:TRACE_CAMERA_SOURCE
```

TRACE does not deliberately log source URLs. Native OpenCV/FFmpeg diagnostics
can contain connection details, so inspect them before sharing logs.

For a run without a preview window:

```powershell
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --headless --max-frames 100 --snapshot storage/preview.jpg
```

Snapshots are saved only when explicitly requested. Nothing records footage
automatically in Phase 1.

## Read the code in this order

| File | Responsibility |
| --- | --- |
| `main.py` | Read command-line arguments and start the program. |
| `config/camera.example.json` | Explain the editable settings through a working example. |
| `vision/utils.py` | `TraceUtils`: shared configuration, resizing, labels, and image saving. |
| `vision/camera.py` | `Camera`: connect, read frames, retry, and release the connection. |
| `vision/detector.py` | `Detector`: load YOLO once and return plain detection dictionaries. |
| `vision/tracker.py` | `Tracker`: match detections across frames and return temporary IDs. |
| `vision/zones.py` | `ZoneManager`: validate polygons and assign zone memberships. |
| `edit_zones.py`, `vision/zone_editor.py` | Draw one polygon on a captured camera frame. |
| `vision/worker.py` | Put those pieces together in one simple video loop. |
| `tests/test_video_input.py` | Check decoding, recovery, cleanup, and validation. |
| `tests/test_tracking.py` | Check real ByteTrack continuity, missed detections, and resets. |
| `docs/BUILD_GUIDE.md` | Explain the pipeline and the remaining build order. |

Functions have descriptive names and docstrings. Related functions are grouped
under labeled comment blocks. Shared operations belong in `TraceUtils`; camera
connection state belongs in `Camera`.

## Phase 2: person detection

Install the updated requirements, then add `--detect` to your working command:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --detect
```

Your existing camera config still works. The default settings are:

```json
"detection": {
  "model": "yolov8n.pt",
  "confidence": 0.4,
  "image_size": 640,
  "device": "cpu",
  "classes": [0]
}
```

Optionally add this object as another field in your camera JSON to change those
settings. `classes: [0]` selects people for this pretrained model. The first
detection run downloads model weights into the working folder and needs internet;
later runs reuse that file. You can also supply a local weights path.

YOLOv8n is a small, established pretrained starting point for the student MVP.
No custom training is required. CPU is the portable default; inference can make
playback slower than recorded speed. A supported NVIDIA GPU with CUDA-enabled
PyTorch can use `"device": "0"`. Actual FPS depends on hardware and footage;
this project does not promise a particular inference rate. Try `image_size: 320`
for a smaller inference input, with potentially lower small-person accuracy.

`Detector.detect(frame)` returns data like this:

```python
[{"class": "person", "class_id": 0, "confidence": 0.91, "bbox": [120.0, 80.0, 240.0, 400.0]}]
```

Boxes use original-frame pixel coordinates. Shared utilities draw on a copy before
resizing the preview. Each box displays `person` and a confidence percentage.
These are detections; persistent track IDs are Phase 3.

The implementation uses the documented
[Ultralytics prediction interface](https://docs.ultralytics.com/modes/predict/).
Verify it on your footage: visible people should receive aligned boxes, empty
scenes should generally have none, and Q should still stop the preview. Try a
higher confidence threshold if you see too many false detections.

## Phase 3: tracking IDs

Add `--track` to enable both YOLO and ByteTrack:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --track
```

You do not need `--detect` as well. Your existing camera configuration works.
The preview now labels people as `person #1 91%`, `person #2 87%`, and so on.
IDs are temporary labels within this camera session; they do not identify people.
Short missed detections can keep the same ID. Long absences and difficult
occlusions can produce a new ID or a mistaken match. New arrivals may need a
second detection before their ID is shown.

The code follows this flow:

1. `Detector.detect()` returns boxes, class IDs, and confidence scores.
2. `Tracker.convert_detections()` converts those dictionaries to NumPy-backed
   Ultralytics `Boxes`, preserving original-frame coordinates.
3. `Tracker.update()` passes them to ByteTrack on **every frame**, including empty
   frames, and returns only the currently visible, confirmed tracks.
4. `TraceUtils.draw_detections()` draws the tracks and their IDs using the same
   drawing function as detection-only mode.

The adapter uses the documented
[Ultralytics ByteTrack interface](https://docs.ultralytics.com/reference/trackers/byte_tracker/).
Ultralytics is pinned to the tested version because this adapter uses its tracker
API directly. `lap` supplies the matching routine; ByteTrack needs no extra model.

Optional tracking settings, also shown in `config/camera.example.json`:

```json
"tracking": {
  "track_high_thresh": 0.4,
  "track_low_thresh": 0.1,
  "new_track_thresh": 0.4,
  "track_buffer_seconds": 5,
  "match_thresh": 0.8,
  "fuse_score": true
}
```

| Setting | Meaning |
| --- | --- |
| `track_high_thresh` | Confidence used in the first matching pass. |
| `track_low_thresh` | Weaker detections above this threshold may continue an existing track. |
| `new_track_thresh` | Minimum confidence to start a new track. |
| `track_buffer_seconds` | Missing-track buffer expressed in seconds of source frames; default 5. |
| `match_thresh` | Maximum matching cost accepted in the first pass; larger is more permissive. |
| `fuse_score` | Include detection confidence when comparing matches in the first pass. |

Tracking mode uses `track_low_thresh` for YOLO's confidence cutoff so it does not
discard ByteTrack's weaker candidates. Detection-only mode still uses
`detection.confidence`. Keep the default person class `[0]` for this phase; tracking
multiple object classes with class-specific matching is not implemented.

The buffer converts to frames using the camera's reported FPS (or `fallback_fps`
if unavailable). At 30 FPS, five seconds means 150 processed frames. It is an
approximate duration: if live inference processes fewer frames per second, those
150 frames take longer than five wall-clock seconds. Existing configs with only
`track_buffer` still use that explicit frame count; `track_buffer_seconds` takes
precedence if both are supplied.

Leaving the field of view can still produce a new ID, even within the buffer.
ByteTrack compares position and predicted motion, not who a person is. A return
near the predicted box can reconnect; a return at a different edge or after a long
absence may not. Increasing the buffer gives more time to match, but cannot
guarantee re-entry identity and can increase mistaken matches between people.

Each output track has `track_id`, `class`, `class_id`, `bbox`, `confidence`,
`first_seen`, `last_seen`, and `current_zone`. Times are seconds: estimated video
time from frame count/FPS for files, or monotonic time for live cameras.
`first_seen` is the first confirmed observation. File timing assumes constant FPS.
`current_zone` contains the first matching zone ID, or `None` when outside zones.

`Camera.connection_id` changes when a connection opens successfully. The worker
uses it to reset tracking after a reconnect. Displayed IDs keep increasing within
the run, so an old ID is not reused after that reset. Metadata is discarded once
ByteTrack forgets a track. Restarting the program starts a new ID session.

Manual success check: walk through the view and confirm your ID stays the same
across consecutive frames. Check two people, a brief occlusion, and a long exit.
Press Q to stop. Use `--headless --max-frames 100 --snapshot storage/tracking.jpg`
for a bounded run without a preview window.

## Phase 4: polygon zones

Stop the current camera preview first, then open the zone editor:

```powershell
.\.venv\Scripts\python.exe edit_zones.py --config config/camera.local.json
```

The editor is a separate window from monitoring; clicks in `main.py --track` do
not create zones. The editor captures one frame and releases the camera. Click at least three
corners around the floor area you want to monitor, in clockwise or anticlockwise
order. The editor closes the polygon for you; clicking near the first corner
again is also accepted.

- **Save button / S / Enter:** save the zone and close the editor.
- **Undo button / Backspace / right-click:** undo the last corner.
- **Clear button / R:** clear this polygon so you can redraw it.
- **Cancel button / Q / Escape / close window:** cancel without saving.

Buttons appear below the picture. Errors are shown below the buttons and in the
terminal. If the camera cannot open, close the monitoring preview and other apps
using that webcam before retrying. Keyboard shortcuts work while the editor window
has focus; mouse buttons do not require keyboard focus.

Then run monitoring:

```powershell
.\.venv\Scripts\python.exe main.py --config config/camera.local.json --track
```

Zone outlines and occupant counts appear on the image. Each person's label shows
their zone names or `Outside zones`. A small dot at the bottom-center of their box
shows the point used for membership. A foot point on the polygon boundary counts
as inside. A shoulder or arm overlapping a boundary alone does not count.

The membership check uses
[OpenCV's point-in-polygon test](https://docs.opencv.org/4.10.0/dc/d48/tutorial_point_polygon_test.html).
This convention is intended for floor regions; perspective, truncated boxes, and
detector errors can still affect the result. Phase 5 will add persistence before
creating alerts. The red box in this phase means zone membership, not an incident.

To add a second zone, use a different ID. To edit an existing one, reuse its ID:

```powershell
.\.venv\Scripts\python.exe edit_zones.py --config config/camera.local.json --zone-id waiting_01 --name "Waiting Area" --type loitering
.\.venv\Scripts\python.exe edit_zones.py --config config/camera.local.json --zone-id ignored_01 --name "Ignore Area" --type ignore
```

Supported types are `restricted`, `loitering`, `crowd`, and `ignore`. Existing names
and types are preserved if omitted while editing. The default zone ID is
`restricted_01`. To delete a zone, remove its object from the camera JSON's `zones`
list and restart monitoring. Zone changes are loaded at startup.

Zones are stored inside each camera's JSON. Example field:

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

Coordinates are **fractions from 0 to 1**, not pixels: `[0.5, 0.5]` is the center
of the image. The editor converts preview clicks into these fractions; membership
and drawing convert them back to original-frame coordinates. This handles
resolution changes of the same camera view. Redraw zones if the view is moved or
cropped. Rectangles and non-crossing concave polygons are supported.

A track can belong to several active zones. It receives `zone_ids`, `zone_names`,
`zone_types`, `current_zone`, and `ignored`. If any ignore zone contains the foot
point, only ignore memberships remain, its box becomes gray, and it is excluded
from active-zone counts. Ignore-zone counts themselves still show ignored people.
The tracker keeps running so IDs do not reset merely from entering an ignore zone.

Running without `--detect` or `--track` shows only the saved zone outlines. Running
with `--detect` shows per-frame membership; use `--track` for ID-based monitoring.
No intrusion, loitering, or crowd incidents are generated in Phase 4.

## Settings

| Setting | Meaning |
| --- | --- |
| `camera_id` | Internal camera label, such as `camera_01`. |
| `name` | Human-readable name reserved for later dashboard use. |
| `source` | Video path, webcam index string, or RTSP URL. |
| `display_width` | Preview width in pixels; aspect ratio is preserved. |
| `fallback_fps` | File playback speed when the source reports no usable FPS. |
| `reconnect_attempts` | Number of attempts after losing a live stream; zero disables retries. |
| `reconnect_delay_seconds` | Delay between live reconnection attempts. |
| `timeout_ms` | RTSP connection and read timeout for the FFmpeg backend. |

RTSP timeouts are passed while opening the stream, as required by
[OpenCV's video I/O documentation](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html).
Webcam driver calls are not covered by those FFmpeg timeouts. OpenCV returns the
same failed-read signal for file EOF and some decode errors; the program reports
that ambiguity rather than claiming it can distinguish them.

## Check it

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests generate temporary video, decode it with OpenCV, and simulate live
disconnections. No camera, model download, or database is needed for these checks.
Before Phase 2, manually confirm that your real camera plays reliably, Q closes
the preview, and disconnecting/reconnecting the stream behaves as expected.

Phases 1–3 were checked with Python 3.12, OpenCV 4.14, Ultralytics 8.4.173, and
lap 0.5.13. All 36 automated tests passed, including polygon geometry, editor saves,
ignore precedence, and a matching return after five
seconds with an eight-second buffer. A 10-frame sample clip also passed
through the real YOLO + ByteTrack command-line pipeline; see the labeled snapshot
at `storage/tracking-preview.jpg`. The clip used translated copies of the bundled
bus image, so it checks integration rather than real-world walking or occlusion
accuracy. Your Phase 1 video playback was manually confirmed. Stable IDs on your
own camera still need a visual check.

Phase 4 also passed a four-frame YOLO + ByteTrack + zones integration check.
The sample's two active occupants and one ignored occupant are shown in
`storage/zones-preview.jpg`. The editor's mouse conversion, saving, and cancellation
were checked automatically; drawing a polygon on your webcam still needs the
manual check above.
