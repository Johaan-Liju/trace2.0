"""Check entry-triggered clips, real FFmpeg output, and recorder cleanup."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from services.clip_service import ClipRecorder
from services.utils import Utils
from services.video_service import run_video


class ClipTests(unittest.TestCase):
    """Only entry events create clips; completed files must be playable."""

    # ---------- Test setup ----------

    def setUp(self):
        """Use a small frame and an isolated output folder with spaces in its path."""
        self.folder = tempfile.TemporaryDirectory(prefix="trace clip test ")
        self.addCleanup(self.folder.cleanup)
        self.config = Utils.load_camera_config("config/camera.example.json")
        self.config["display_width"] = 160
        self.config["clips"] = {"output_dir": self.folder.name, "duration_seconds": 1, "fps": 10}
        self.frame = np.full((91, 160, 3), 90, dtype=np.uint8)
        self.person = {"track_id": 1, "class": "person"}

    def recorder(self):
        """Use a fake executable path for tests that mock encoding."""
        with patch("services.utils.shutil.which", return_value="ffmpeg"):
            return ClipRecorder(self.config)

    # ---------- Entry selection ----------

    def test_empty_frames_and_other_classes_create_no_files(self):
        """An empty view or a vehicle does not start recording."""
        recorder = self.recorder()
        recorder.update(self.frame, [], set(), 0)
        recorder.update(self.frame, [{"track_id": 2, "class": "car"}], {2}, 1)
        self.assertIsNone(recorder.close())
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])

    def test_staying_inside_and_brief_tracking_gap_do_not_repeat(self):
        """A continuous person produces one clip, including after a missed frame."""
        recorder = self.recorder()
        with patch.object(Utils, "encode_clip", return_value=Path("entry.mp4")) as encode:
            recorder.update(self.frame, [self.person], {1}, 0)
            recorder.update(self.frame, [], {1}, 0.5)
            self.assertEqual(recorder.update(self.frame, [self.person], {1}, 1), Path("entry.mp4"))
            recorder.update(self.frame, [self.person], {1}, 2)
            self.assertIsNone(recorder.close())
            encode.assert_called_once()

    def test_new_people_during_a_clip_share_it_and_later_entries_start_another(self):
        """Merge simultaneous arrivals without creating duplicate video files."""
        recorder = self.recorder()
        with patch.object(Utils, "encode_clip", return_value=Path("entry.mp4")) as encode:
            recorder.update(self.frame, [self.person], {1}, 0)
            second = dict(self.person, track_id=2)
            recorder.update(self.frame, [self.person, second], {1, 2}, 0.5)
            recorder.update(self.frame, [self.person, second], {1, 2}, 1)
            recorder.update(self.frame, [], set(), 2)
            recorder.update(self.frame, [dict(self.person, track_id=3)], {3}, 3)
            recorder.close()
            self.assertEqual(encode.call_count, 2)

    def test_reset_finishes_clip_and_rearms_entry_detection(self):
        """Reconnection finishes the old clip and allows a fresh observation."""
        recorder = self.recorder()
        with patch.object(Utils, "encode_clip", return_value=Path("entry.mp4")) as encode:
            recorder.update(self.frame, [self.person], {1}, 0)
            recorder.reset()
            recorder.update(self.frame, [self.person], {1}, 1)
            recorder.close()
            self.assertEqual(encode.call_count, 2)

    def test_return_after_absence_records_even_with_the_same_tracking_id(self):
        """A remembered person returning after two seconds can produce a new clip."""
        recorder = self.recorder()
        with patch.object(Utils, "encode_clip", return_value=Path("entry.mp4")) as encode:
            recorder.update(self.frame, [self.person], {1}, 0)
            recorder.update(self.frame, [], {1}, 1)
            recorder.update(self.frame, [], {1}, 2)
            recorder.update(self.frame, [self.person], {1}, 2.5)
            recorder.close()
            self.assertEqual(encode.call_count, 2)

    # ---------- Encoding and cleanup ----------

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the real encoder check")
    def test_real_mp4_preserves_elapsed_time_and_is_readable(self):
        """Irregular detections encode one second of video, with even output dimensions."""
        recorder = ClipRecorder(self.config)
        try:
            recorder.update(self.frame, [self.person], {1}, 0)
            recorder.update(np.full_like(self.frame, 160), [self.person], {1}, 0.3)
            recorder.update(np.full_like(self.frame, 220), [self.person], {1}, 0.8)
            temporary_folder = Path(recorder.folder.name)
            path = recorder.update(self.frame, [self.person], {1}, 1)
            self.assertTrue(path.is_file())
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
                self.assertAlmostEqual(duration, 1, delta=0.1)
                frames = []
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    frames.append(frame)
                self.assertGreaterEqual(len(frames), 9)
                self.assertEqual(frames[0].shape[:2], (92, 160))
                self.assertGreater(frames[-1].mean(), frames[0].mean() + 80)
            finally:
                capture.release()
            self.assertFalse(temporary_folder.exists())
            self.assertEqual(len(list(Path(self.folder.name).glob("*.mp4"))), 1)
        finally:
            recorder.close()

    def test_failed_encoding_cleans_temporary_frames(self):
        """A failed encoder must not leave temporary images or report a ready clip."""
        recorder = self.recorder()
        recorder.update(self.frame, [self.person], {1}, 0)
        temporary_folder = Path(recorder.folder.name)
        with patch.object(Utils, "encode_clip", side_effect=RuntimeError("Encoder failed")):
            with self.assertRaises(RuntimeError):
                recorder.close()
        self.assertFalse(temporary_folder.exists())
        self.assertIsNone(recorder.close())

    def test_timeout_removes_partial_output(self):
        """A timed-out FFmpeg run never publishes a completed MP4."""
        folder = Path(self.folder.name)
        output = folder / "entry.mp4"
        output.with_suffix(".partial.mp4").write_bytes(b"unfinished")
        settings = {"ffmpeg": "ffmpeg", "fps": 10}
        with patch("services.utils.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 60)):
            with self.assertRaisesRegex(RuntimeError, "too long"):
                Utils.encode_clip(folder, [("frame_000000.jpg", 0)], 1, output, settings)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".partial.mp4").exists())

    # ---------- Configuration and monitoring integration ----------

    def test_invalid_settings_and_missing_ffmpeg_fail_clearly(self):
        """Reject invalid timing settings and report a missing encoder before capture."""
        for supplied in (None, {"duration_seconds": 0}, {"duration_seconds": float("nan")},
                         {"fps": True}, {"fps": 100}, {"output_dir": ""}):
            with self.subTest(settings=supplied), self.assertRaises(ValueError):
                Utils.get_clip_config({"clips": supplied})
        with patch("services.utils.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg is missing"):
                Utils.get_clip_config({})

    def test_clips_enable_tracking_without_requiring_zones(self):
        """--clips alone enables detection and tracking, then finalizes on EOF."""
        with patch("services.video_service.Camera") as camera_class, \
                patch("services.video_service.Detector"), \
                patch("services.video_service.Tracker") as tracker_class, \
                patch("services.video_service.ClipRecorder") as recorder_class:
            camera = camera_class.return_value
            camera.get_fps.return_value = 25
            camera.read_frame.side_effect = [self.frame, None]
            tracker_class.return_value.update.return_value = []
            self.assertEqual(run_video(self.config, headless=True, clips=True), 1)
            tracker_class.assert_called_once_with(self.config, frame_rate=25)
            recorder_class.return_value.update.assert_called_once()
            recorder_class.return_value.close.assert_called_once()
            camera.close.assert_called_once()

    def test_processing_error_still_finalizes_clip_and_releases_camera(self):
        """Keep a usable partial-length clip when later processing fails."""
        with patch("services.video_service.Camera") as camera_class, \
                patch("services.video_service.Detector") as detector_class, \
                patch("services.video_service.Tracker"), \
                patch("services.video_service.ClipRecorder") as recorder_class:
            camera_class.return_value.get_fps.return_value = 25
            camera_class.return_value.read_frame.return_value = self.frame
            detector_class.return_value.detect.side_effect = RuntimeError("Inference failed")
            with self.assertRaisesRegex(RuntimeError, "Inference failed"):
                run_video(self.config, headless=True, clips=True)
            recorder_class.return_value.close.assert_called_once()
            camera_class.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
