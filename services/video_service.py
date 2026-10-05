"""Video service: the main loop, camera input, YOLO, and ByteTrack.

Read run_video() first, then process_frame(). The classes below hold only the
state that must survive between frames: the camera, model, and tracked IDs.
"""

import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from services.utils import Utils
from services.event_service import EntryAlerts


# ==================== 1. RUN THE VIDEO ====================

def run_video(config, headless=False, max_frames=None, snapshot_path=None, detect=False, track=False, alerts=False):
    """Read video until EOF, Q, Ctrl+C, or the requested frame limit."""
    camera = Camera(config)
    frame_number = 0

    try:
        zones = Utils.get_zones(config)
        entry_alerts = EntryAlerts(config, zones) if alerts else None
        track = track or alerts
        detector = Detector(config) if detect or track else None
        if not camera.open():
            raise ConnectionError("Cannot open camera. Check the source and connection.")

        frame_rate = camera.get_fps()
        frame_interval = 1 / frame_rate
        tracker = Tracker(config, frame_rate=frame_rate) if track else None
        if tracker is not None:
            logging.info("Tracking remembers missing people for up to %s processed frames.",
                         tracker.settings["track_buffer"])
        connection_id = camera.connection_id
        logging.info("Camera online. Press Q in the preview or Ctrl+C to stop.")

        while True:
            frame_started = time.monotonic()
            frame = camera.read_frame()
            if frame is None:
                if frame_number == 0:
                    raise RuntimeError("Video opened but no frames could be decoded.")
                logging.info("Video ended or decoder stopped after %s frames.", frame_number)
                break

            frame_number += 1

            if tracker is not None and camera.connection_id != connection_id:
                tracker.reset()
                if entry_alerts is not None:
                    entry_alerts.reset()
                connection_id = camera.connection_id
                logging.info("Camera reconnected. Starting fresh tracking IDs.")

            # Follow the source timeline, then process and draw this frame.
            timestamp = time.monotonic()
            if camera.source_type == "file":
                timestamp = (frame_number - 1) * frame_interval
            annotated = process_frame(frame, detector, tracker, zones, timestamp, entry_alerts)

            preview = Utils.resize_frame(annotated, config["display_width"])
            preview = Utils.add_preview_label(
                preview, config["camera_id"], frame_number
            )
            if entry_alerts is not None and entry_alerts.last_message:
                preview = Utils.draw_alert_banner(preview, entry_alerts.last_message)

            if snapshot_path and frame_number == 1:
                Utils.save_snapshot(preview, snapshot_path)

            if not headless:
                cv2.imshow("TRACE - Camera Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("TRACE - Camera Preview", cv2.WND_PROP_VISIBLE) < 1:
                    break

            if max_frames is not None and frame_number >= max_frames:
                break

            # Local video files decode quickly; pace them at their recorded FPS.
            if camera.source_type == "file":
                remaining = frame_interval - (time.monotonic() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)

        return frame_number

    finally:
        camera.close()
        if not headless:
            cv2.destroyAllWindows()
        logging.info("Camera released.")


# ==================== 2. PROCESS ONE FRAME ====================

def process_frame(frame, detector, tracker, zones, timestamp, entry_alerts=None):
    """Detect people, attach IDs and zones, then draw the result."""
    objects = []
    if tracker is not None:
        detections = detector.detect(frame, confidence=tracker.settings["track_low_thresh"])
        objects = tracker.update(detections, frame.shape, timestamp)
    elif detector is not None:
        objects = detector.detect(frame)

    annotated = frame
    if zones:
        objects = Utils.assign_zones(objects, zones, frame.shape)
        counts = Utils.count_zone_occupants(objects, zones) if detector is not None else None
        annotated = Utils.draw_zones(frame, zones, counts)
    if entry_alerts is not None:
        entries = entry_alerts.check_entries(objects, tracker.get_remembered_ids())
        entry_alerts.notify(entries)
    if detector is not None:
        annotated = Utils.draw_detections(annotated, objects)
    return annotated


# ==================== 3. CAMERA INPUT ====================

class Camera:
    """Read one local video, webcam, or RTSP stream with OpenCV."""

    # ---------- Camera setup ----------

    def __init__(self, config):
        """Store settings; the connection starts when open() is called."""
        self.config = config
        self.source = config["source"]
        self.source_type = Utils.get_source_type(self.source)
        self.capture = None
        self.status = "OFFLINE"
        self.connection_id = 0

    def open(self):
        """Open the source and return True if OpenCV can read from it."""
        self.close()
        if self.source_type == "file" and not Path(self.source).is_file():
            raise FileNotFoundError("Video file does not exist. Check the camera source.")

        if self.source_type == "rtsp":
            # OpenCV requires these timeouts when opening the FFmpeg capture.
            options = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config["timeout_ms"],
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config["timeout_ms"],
            ]
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, options)
        elif self.source_type == "webcam":
            self.capture = cv2.VideoCapture(int(self.source))
        else:
            self.capture = cv2.VideoCapture(self.source)

        if self.capture.isOpened():
            self.status = "ONLINE"
            self.connection_id += 1
            return True

        self.close()
        return False

    # ---------- Frame reading and reconnection ----------

    def read_frame(self):
        """Return a frame, None at file end, or an error after live retries."""
        if self.capture is None:
            raise RuntimeError("Open the camera before reading a frame.")

        success, frame = self.capture.read()
        if success:
            return frame

        if self.source_type == "file":
            # OpenCV uses the same result for EOF and a decode failure.
            self.status = "ENDED"
            return None

        self.status = "OFFLINE"
        for attempt in range(self.config["reconnect_attempts"]):
            logging.warning("Camera offline. Reconnection attempt %s of %s.",
                            attempt + 1, self.config["reconnect_attempts"])
            time.sleep(self.config["reconnect_delay_seconds"])
            if self.open():
                success, frame = self.capture.read()
                if success:
                    return frame
                self.status = "OFFLINE"

        self.close()
        raise ConnectionError("Camera disconnected and reconnection attempts failed.")

    def get_fps(self):
        """Get playback speed, with a fallback for missing camera metadata."""
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            return self.config["fallback_fps"]
        return fps

    # ---------- Resource cleanup ----------

    def close(self):
        """Release the camera handle, including after an error."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status = "OFFLINE"


# ==================== 4. PERSON DETECTION ====================

class Detector:
    """Load one pretrained YOLO model and reuse it for every frame."""

    # ---------- Model setup ----------

    def __init__(self, config):
        """Validate settings and load the model once, before reading video."""
        self.settings = Utils.get_detection_config(config)
        self.model = self.load_model()
        if self.model.task != "detect":
            raise ValueError("Choose an object-detection model, such as yolov8n.pt.")
        for class_id in self.settings["classes"]:
            if class_id not in self.model.names:
                raise ValueError(f"Class ID {class_id} is not available in this model.")

    def load_model(self):
        """Import YOLO only when needed; first use may download its weights."""
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO dependencies are missing. Run: python -m pip install -r requirements.txt"
            ) from error

        logging.info("Loading YOLO. First use may download pretrained weights.")
        try:
            return YOLO(self.settings["model"])
        except Exception as error:
            raise RuntimeError(
                "Cannot load the YOLO model. Check its path, download connection, "
                "and installed dependencies."
            ) from error

    # ---------- Frame detection ----------

    def detect(self, frame, confidence=None):
        """Return class, confidence, and [x1, y1, x2, y2] for each detection."""
        # Tracking keeps weaker detections to help match existing people.
        if confidence is None:
            confidence = self.settings["confidence"]
        result = self.model.predict(
            source=frame,
            conf=confidence,
            imgsz=self.settings["image_size"],
            device=self.settings["device"],
            classes=self.settings["classes"],
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        # Ultralytics returns one row per box: x1, y1, x2, y2, score, class ID.
        for row in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, confidence, class_id = row
            detections.append({
                "class": result.names[int(class_id)],
                "class_id": int(class_id),
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        return detections


# ==================== 5. PERSON TRACKING ====================

class Tracker:
    """Keep one ByteTrack instance and its temporary IDs for one camera."""

    # ---------- Tracker setup ----------

    def __init__(self, config, frame_rate=30):
        """Load ByteTrack without downloading another model."""
        self.settings = Utils.get_tracking_config(config, frame_rate)
        try:
            # Check lap explicitly so Ultralytics does not try to auto-install it.
            import lap
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as error:
            raise RuntimeError(
                "Tracking dependencies are missing. Run: python -m pip install -r requirements.txt"
            ) from error

        self.boxes_class = Boxes
        self.engine = BYTETracker(SimpleNamespace(**self.settings))
        self.track_history = {}
        self.next_track_id = 1

    # ---------- Detection conversion ----------

    def convert_detections(self, detections, frame_shape):
        """Convert dictionaries to the NumPy-backed Boxes ByteTrack expects."""
        rows = []
        for detection in detections:
            rows.append([
                *detection["bbox"], detection["confidence"], detection["class_id"]
            ])

        # An empty frame still needs shape (0, 6), so lost tracks can age.
        data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        return self.boxes_class(data, orig_shape=frame_shape[:2])

    # ---------- Tracking and output ----------

    def update(self, detections, frame_shape, timestamp):
        """Return visible tracks; timestamp is seconds on the source timeline."""
        boxes = self.convert_detections(detections, frame_shape)
        results = self.engine.update(boxes)
        tracks = []

        for row in results:
            x1, y1, x2, y2, internal_id, confidence, class_id, detection_index = row
            internal_id = int(internal_id)
            detection = detections[int(detection_index)]

            if internal_id not in self.track_history:
                self.track_history[internal_id] = {
                    "track_id": self.next_track_id,
                    "first_seen": timestamp,
                }
                self.next_track_id += 1

            history = self.track_history[internal_id]
            tracks.append({
                "track_id": history["track_id"],
                "class": detection["class"],
                "class_id": int(class_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
                "first_seen": history["first_seen"],
                "last_seen": timestamp,
                "current_zone": None,
            })

        self.remove_expired_history()
        return tracks

    # ---------- State cleanup ----------

    def get_remembered_ids(self):
        """Return visible and briefly lost IDs so missed frames do not re-alert."""
        return {history["track_id"] for history in self.track_history.values()}

    def remove_expired_history(self):
        """Forget metadata once ByteTrack no longer keeps the track alive."""
        remembered_tracks = self.engine.tracked_stracks + self.engine.lost_stracks
        remembered_ids = {track.track_id for track in remembered_tracks}
        self.track_history = {
            track_id: history for track_id, history in self.track_history.items()
            if track_id in remembered_ids
        }

    def reset(self):
        """Clear stale matches after reconnection; do not reuse displayed IDs."""
        self.engine.reset()
        self.track_history.clear()
