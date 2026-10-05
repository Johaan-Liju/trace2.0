"""Exercise real ByteTrack with controlled boxes; no model or video download."""

import unittest
from unittest.mock import patch

import numpy as np

from vision.tracker import Tracker
from vision.utils import TraceUtils
from vision.worker import run_video


class TrackingTests(unittest.TestCase):
    """Check stable IDs, missed detections, weak detections, and resets."""

    # ---------- Test setup ----------

    def setUp(self):
        """Give each test its own tracker with a short lost-track buffer."""
        self.tracker = Tracker({"tracking": {"track_buffer": 3}})
        self.frame_shape = (480, 640, 3)

    def person(self, x=20, confidence=0.9):
        """Build a person detection in original-frame pixel coordinates."""
        return {"class": "person", "class_id": 0, "confidence": confidence,
                "bbox": [x, 50, x + 60, 200]}

    def update(self, detections, timestamp):
        """Feed a synthetic frame to the real tracker."""
        return self.tracker.update(detections, self.frame_shape, timestamp)

    # ---------- ID continuity checks ----------

    def test_moving_people_keep_ids_when_detection_order_changes(self):
        """Match positions over time instead of assigning IDs by list order."""
        first = self.update([self.person(20), self.person(300)], 0.0)
        second = self.update([self.person(303), self.person(23)], 0.1)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        first.sort(key=lambda track: track["bbox"][0])
        second.sort(key=lambda track: track["bbox"][0])
        self.assertEqual([track["track_id"] for track in first],
                         [track["track_id"] for track in second])
        self.assertNotEqual(first[0]["track_id"], first[1]["track_id"])
        self.assertEqual(second[0]["first_seen"], 0.0)
        self.assertEqual(second[0]["last_seen"], 0.1)
        self.assertIsNone(second[0]["current_zone"])

    def test_short_missing_detection_recovers_same_id(self):
        """Do not draw missing people, but remember their IDs briefly."""
        first = self.update([self.person()], 0.0)[0]
        self.assertEqual(self.update([], 0.1), [])
        returned = self.update([self.person(22)], 0.2)[0]
        self.assertEqual(returned["track_id"], first["track_id"])

    def test_weak_detection_continues_but_does_not_start_a_track(self):
        """ByteTrack's second pass rescues an existing low-confidence match."""
        self.assertEqual(self.update([self.person(confidence=0.2)], 0.0), [])
        self.update([self.person()], 0.1)
        confirmed = self.update([self.person()], 0.2)[0]
        weak = self.update([self.person(22, confidence=0.2)], 0.3)[0]
        self.assertEqual(weak["track_id"], confirmed["track_id"])
        self.assertAlmostEqual(weak["confidence"], 0.2)

    def test_expired_track_gets_new_id_and_history_is_removed(self):
        """An absence beyond the buffer cannot keep the previous ID forever."""
        first = self.update([self.person()], 0.0)[0]
        for frame_number in range(1, 8):
            self.assertEqual(self.update([], frame_number / 10), [])
        self.assertEqual(self.tracker.track_history, {})
        self.update([self.person()], 0.8)
        returned = self.update([self.person()], 0.9)[0]
        self.assertNotEqual(returned["track_id"], first["track_id"])

    def test_reset_does_not_reuse_displayed_ids(self):
        """A reconnect starts fresh IDs even if internal ByteTrack IDs reset."""
        first = self.update([self.person()], 0.0)[0]
        self.tracker.reset()
        second = self.update([self.person()], 5.0)[0]
        self.assertNotEqual(second["track_id"], first["track_id"])
        self.assertEqual(second["first_seen"], 5.0)

    # ---------- Configuration and pipeline checks ----------

    def test_buffer_seconds_uses_camera_frame_rate(self):
        """Five seconds maps to the source FPS, with legacy frame support."""
        self.assertEqual(TraceUtils.get_tracking_config({}, frame_rate=30)["track_buffer"], 150)
        self.assertEqual(TraceUtils.get_tracking_config({}, frame_rate=15)["track_buffer"], 75)
        settings = {"tracking": {"track_buffer_seconds": 2, "track_buffer": 30}}
        self.assertEqual(TraceUtils.get_tracking_config(settings, frame_rate=25)["track_buffer"], 50)
        self.assertEqual(TraceUtils.get_tracking_config({"tracking": {"track_buffer": 7}})["track_buffer"], 7)

    def test_default_buffer_recovers_matching_return_after_two_seconds(self):
        """A matching return can survive more than the old 30-frame buffer."""
        self.tracker = Tracker({}, frame_rate=30)
        first = self.update([self.person()], 0.0)[0]
        for frame_number in range(1, 61):
            self.assertEqual(self.update([], frame_number / 30), [])
        returned = self.update([self.person()], 61 / 30)[0]
        self.assertEqual(returned["track_id"], first["track_id"])

    def test_distant_return_is_not_forced_into_an_old_id(self):
        """A longer buffer does not identify somebody at an unrelated position."""
        self.tracker = Tracker({}, frame_rate=30)
        first = self.update([self.person(20)], 0.0)[0]
        self.update([], 0.1)
        self.update([self.person(400)], 0.2)
        returned = self.update([self.person(400)], 0.3)[0]
        self.assertNotEqual(returned["track_id"], first["track_id"])

    def test_eight_second_buffer_can_recover_a_five_second_gap(self):
        """Cover the configured 2–5 second exit with a matching return."""
        self.tracker = Tracker({"tracking": {"track_buffer_seconds": 8}}, frame_rate=30)
        first = self.update([self.person()], 0.0)[0]
        for frame_number in range(1, 151):
            self.update([], frame_number / 30)
        returned = self.update([self.person()], 151 / 30)[0]
        self.assertEqual(returned["track_id"], first["track_id"])

    def test_invalid_tracking_settings(self):
        """Reject contradictory thresholds and invalid buffer sizes."""
        invalid = [None, {"track_buffer": 0}, {"track_buffer": True},
                   {"track_low_thresh": 0.5}, {"new_track_thresh": 0.2},
                   {"match_thresh": float("nan")}, {"fuse_score": "true"},
                   {"track_buffer_seconds": 0}, {"track_buffer_seconds": True},
                   {"track_buffer_seconds": float("inf")}]
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    TraceUtils.get_tracking_config({"tracking": settings})

    def test_worker_enables_detection_and_resets_after_reconnect(self):
        """The --track path lowers detector confidence and notices reconnects."""
        config = TraceUtils.load_camera_config("config/camera.example.json")
        frame = np.zeros(self.frame_shape, dtype=np.uint8)
        with patch("vision.worker.Camera") as camera_class:
            with patch("vision.worker.Detector") as detector_class:
                with patch("vision.worker.Tracker") as tracker_class:
                    camera = camera_class.return_value
                    camera.get_fps.return_value = 25
                    camera.source_type = "webcam"
                    camera.connection_id = 1

                    def reconnect_and_read():
                        """Simulate a successful reconnection inside read_frame."""
                        camera.connection_id = 2
                        return frame

                    camera.read_frame.side_effect = reconnect_and_read
                    detector_class.return_value.detect.return_value = []
                    tracker = tracker_class.return_value
                    tracker.settings = TraceUtils.get_tracking_config(config)
                    tracker.update.return_value = []
                    self.assertEqual(run_video(config, True, 1, track=True), 1)
                    tracker_class.assert_called_once_with(config, frame_rate=25)
                    tracker.reset.assert_called_once()
                    self.assertEqual(detector_class.return_value.detect.call_args.kwargs,
                                     {"confidence": 0.1})
                    tracker.update.assert_called_once()
                    camera.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
