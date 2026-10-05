"""Phase 3: associate YOLO detections across frames with ByteTrack."""

from types import SimpleNamespace

import numpy as np

from vision.utils import TraceUtils


class Tracker:
    """Keep one ByteTrack instance and its temporary IDs for one camera."""

    # ---------- Tracker setup ----------

    def __init__(self, config, frame_rate=30):
        """Load ByteTrack without downloading another model."""
        self.settings = TraceUtils.get_tracking_config(config, frame_rate)
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
