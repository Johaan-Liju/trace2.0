"""Run with: python -m unittest discover -s tests -v."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from services.video_service import Camera
from services.utils import Utils
from services.video_service import run_video


class VideoInputTests(unittest.TestCase):
    """Check actual file decoding and simulated camera failure paths."""

    # ---------- Test setup ----------

    def setUp(self):
        """Create an isolated folder and valid camera settings."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.config = Utils.load_camera_config("config/camera.example.json")
        self.config["source"] = str(self.root / "sample.avi")

    def create_video(self, frame_count=5):
        """Write a small real video to exercise OpenCV rather than mock it."""
        writer = cv2.VideoWriter(self.config["source"],
                                 cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
        self.assertTrue(writer.isOpened())
        try:
            for number in range(frame_count):
                frame = np.full((120, 160, 3), number * 30, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    # ---------- Real video checks ----------

    def test_file_playback_and_snapshot(self):
        """Decode to EOF and verify that the requested snapshot is readable."""
        self.create_video()
        snapshot = self.root / "preview.jpg"
        count = run_video(self.config, headless=True, snapshot_path=snapshot)
        self.assertEqual(count, 5)
        self.assertEqual(cv2.imread(str(snapshot)).shape[:2], (720, 960))

    def test_frame_limit(self):
        """A smoke run stops early when a frame limit is supplied."""
        self.create_video()
        self.assertEqual(run_video(self.config, headless=True, max_frames=2), 2)

    def test_missing_file(self):
        """Missing source files produce an actionable error."""
        with self.assertRaises(FileNotFoundError):
            run_video(self.config, headless=True)

    def test_cleanup_after_processing_error(self):
        """Release the camera even when later frame processing fails."""
        with patch("services.video_service.Camera") as camera_class:
            camera = camera_class.return_value
            camera.get_fps.return_value = 25
            camera.read_frame.side_effect = RuntimeError("Processing failed")
            with self.assertRaises(RuntimeError):
                run_video(self.config, headless=True)
            camera.close.assert_called_once()

    # ---------- Live camera recovery checks ----------

    def test_reconnect_recovers(self):
        """A dropped live stream can reopen and return its next frame."""
        self.config["source"] = "rtsp://example.invalid/stream"
        camera = Camera(self.config)
        capture = MagicMock()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        capture.read.side_effect = [(False, None), (True, frame)]
        camera.capture = capture
        with patch.object(camera, "open", return_value=True) as reopen:
            with patch("services.video_service.time.sleep"):
                self.assertIs(camera.read_frame(), frame)
        reopen.assert_called_once()

    def test_reconnect_failure_is_bounded(self):
        """Failed live streams stop after the configured retry count."""
        self.config["source"] = "0"
        camera = Camera(self.config)
        capture = MagicMock()
        capture.read.return_value = (False, None)
        camera.capture = capture
        with patch.object(camera, "open", return_value=False) as reopen:
            with patch("services.video_service.time.sleep"):
                with self.assertRaises(ConnectionError):
                    camera.read_frame()
        self.assertEqual(reopen.call_count, self.config["reconnect_attempts"])
        capture.release.assert_called_once()
        self.assertEqual(camera.status, "OFFLINE")

    # ---------- Configuration checks ----------

    def test_invalid_settings(self):
        """Reject values that would break resizing, timing, or retry loops."""
        cases = [("display_width", 0), ("timeout_ms", 1.5),
                 ("reconnect_attempts", -1), ("fallback_fps", float("nan")),
                 ("name", ""), ("reconnect_delay_seconds", True)]
        for field, value in cases:
            with self.subTest(field=field):
                config = dict(self.config)
                config[field] = value
                path = self.root / "invalid.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    Utils.load_camera_config(path)

    def test_private_source_override(self):
        """Allow credentials to be supplied outside the checked-in JSON."""
        with patch.dict("os.environ", {"TRACE_CAMERA_SOURCE": "0"}):
            config = Utils.load_camera_config("config/camera.example.json")
        self.assertEqual(config["source"], "0")


if __name__ == "__main__":
    unittest.main()
