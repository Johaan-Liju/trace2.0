"""Video input only: opening, reading, reconnecting, and closing a camera."""

import logging
import math
import time
from pathlib import Path

import cv2

from vision.utils import TraceUtils


class Camera:
    """Read one local video, webcam, or RTSP stream with OpenCV."""

    # ---------- Camera setup ----------

    def __init__(self, config):
        """Store settings; the connection starts when open() is called."""
        self.config = config
        self.source = config["source"]
        self.source_type = TraceUtils.get_source_type(self.source)
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
