"""Short FFmpeg clips triggered by a newly tracked person in the camera view."""

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from services.utils import Utils


class ClipRecorder:
    """Keep one short entry clip at a time and return its finished MP4 path."""

    # ---------- Recorder setup ----------

    def __init__(self, config):
        """Validate settings; temporary image storage starts only after an entry."""
        self.settings = Utils.get_clip_config(config)
        self.width = config["display_width"]
        self.output_dir = Path(self.settings["output_dir"]).resolve()
        self.last_seen = {}
        self.folder = None
        self.frames = []
        self.start_time = None
        self.last_time = None

    # ---------- Entry detection and frame collection ----------

    def update(self, frame, tracks, remembered_ids, timestamp):
        """Collect an entry clip; return a Path when a clip finishes, else None."""
        new_ids = self.find_entries(tracks, remembered_ids, timestamp)

        finished_path = None
        if self.folder is not None and timestamp >= self.start_time + self.settings["duration_seconds"]:
            finished_path = self.finish(self.start_time + self.settings["duration_seconds"])
        if new_ids and self.folder is None:
            self.start(timestamp)
        if self.folder is not None:
            self.last_time = timestamp
            if new_ids or not self.frames or timestamp - self.frames[-1][1] >= 1 / self.settings["fps"] - 1e-9:
                self.add_frame(frame, timestamp)
        return finished_path

    def find_entries(self, tracks, remembered_ids, timestamp):
        """Find new people or returns after a gap, allowing brief missed detections."""
        visible_ids = {track["track_id"] for track in tracks if track["class"] == "person"}
        new_ids = set()
        for track_id in visible_ids:
            previous_time = self.last_seen.get(track_id)
            if previous_time is None or timestamp - previous_time >= self.settings["reentry_gap_seconds"]:
                new_ids.add(track_id)
            self.last_seen[track_id] = timestamp
        remembered = set(remembered_ids) | visible_ids
        for track_id in list(self.last_seen):
            if track_id not in remembered:
                del self.last_seen[track_id]
        return new_ids

    def start(self, timestamp):
        """Start a fixed-duration clip at the person's first visible detection."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(prefix="trace-entry-")
        self.frames = []
        self.start_time = timestamp
        self.last_time = timestamp
        logging.info("Person entered the frame. Collecting a %s-second clip.",
                     self.settings["duration_seconds"])

    def add_frame(self, frame, timestamp):
        """Save a sampled camera image at a fixed size, without preview overlays."""
        preview = Utils.resize_frame(frame, self.width)
        if self.frames:
            preview = cv2.resize(preview, self.frame_size)
        else:
            self.frame_size = (preview.shape[1], preview.shape[0])
        name = f"frame_{len(self.frames):06d}.jpg"
        Utils.save_snapshot(preview, Path(self.folder.name) / name)
        self.frames.append((name, timestamp))

    # ---------- Clip output and cleanup ----------

    def finish(self, end_time):
        """Encode collected frames, clean temporary files, and return the MP4."""
        if self.folder is None:
            return None
        try:
            if not self.frames:
                return None
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output_path = self.output_dir / f"entry_{stamp}_{uuid4().hex[:8]}.mp4"
            path = Utils.encode_clip(self.folder.name, self.frames, end_time, output_path, self.settings)
            logging.info("CLIP_READY: %s", path)
            return path
        finally:
            self.folder.cleanup()
            self.folder = None
            self.frames = []

    def close(self):
        """Finish a shorter clip on stop, EOF, failure, or camera reconnection."""
        if self.folder is None:
            return None
        end_time = min(self.last_time + 1 / self.settings["fps"],
                       self.start_time + self.settings["duration_seconds"])
        return self.finish(end_time)

    def reset(self):
        """Finish the old clip and forget IDs when the camera reconnects."""
        path = self.close()
        self.last_seen.clear()
        return path
