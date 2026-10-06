"""Behavior checks for immediate restricted-area entry alerts."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.event_service import EntryAlerts
from services.utils import Utils
from services.video_service import process_frame, run_video


class EntryAlertTests(unittest.TestCase):
    """An entry alerts once, continuous presence does not, and re-entry alerts again."""

    # ---------- Test fixtures ----------

    def setUp(self):
        """Create one restricted zone with sound disabled for automated checks."""
        self.zone = {"id": "door", "name": "Door", "type": "restricted",
                     "points": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}
        self.config = {"camera_id": "camera_01", "alerts": {"sound": False}}
        self.alerts = EntryAlerts(self.config, [self.zone])

    def person(self, track_id=1, inside=True):
        """Build a track after zone membership has been assigned."""
        return {"track_id": track_id, "class": "person", "confidence": 0.9,
                "zone_ids": ["door"] if inside else [], "ignored": False}

    # ---------- Entry rules ----------

    def test_entry_alerts_immediately_and_standing_does_not_repeat(self):
        """No dwell timer is involved: the first inside observation alerts."""
        entries = self.alerts.check_entries([self.person()], {1})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event_type"], "RESTRICTED_ENTRY")
        self.assertEqual(entries[0]["zone_id"], "door")
        for frame in range(5):
            self.assertEqual(self.alerts.check_entries([self.person()], {1}), [])

    def test_leaving_and_returning_triggers_a_second_alert(self):
        """An observed exit rearms the alert without any cooldown delay."""
        self.alerts.check_entries([self.person()], {1})
        self.assertEqual(self.alerts.check_entries([self.person(inside=False)], {1}), [])
        self.assertEqual(len(self.alerts.check_entries([self.person()], {1})), 1)

    def test_brief_missed_detection_does_not_repeat_alert(self):
        """Missing one detection does not prove that somebody left the zone."""
        self.alerts.check_entries([self.person()], {1})
        self.assertEqual(self.alerts.check_entries([], {1}), [])
        self.assertEqual(self.alerts.check_entries([self.person()], {1}), [])

    def test_expired_tracks_are_removed_and_new_person_alerts(self):
        """Do not retain old alert membership after the tracker forgets an ID."""
        self.alerts.check_entries([self.person()], {1})
        self.alerts.check_entries([], set())
        self.assertEqual(self.alerts.inside_zones, {})
        self.assertEqual(len(self.alerts.check_entries([self.person(2)], {2})), 1)

    def test_each_person_and_restricted_zone_gets_its_own_alert(self):
        """Two people and overlapping restricted zones are checked separately."""
        second_zone = dict(self.zone, id="gate", name="Gate")
        alerts = EntryAlerts(self.config, [self.zone, second_zone])
        first = self.person(1)
        first["zone_ids"] = ["door", "gate"]
        events = alerts.check_entries([first, self.person(2)], {1, 2})
        self.assertEqual({(event["track_id"], event["zone_id"]) for event in events},
                         {(1, "door"), (1, "gate"), (2, "door")})

    def test_ignored_people_and_other_classes_do_not_alert(self):
        """Respect ignore zones and alert only for people."""
        ignored = dict(self.person(), ignored=True)
        vehicle = dict(self.person(2), **{"class": "car"})
        self.assertEqual(self.alerts.check_entries([ignored, vehicle], {1, 2}), [])

    def test_reconnect_starts_a_fresh_observation(self):
        """A person present when monitoring restarts produces an alert."""
        self.alerts.check_entries([self.person()], {1})
        self.alerts.reset()
        self.assertEqual(self.alerts.last_message, "")
        self.assertEqual(len(self.alerts.check_entries([self.person()], {1})), 1)

    # ---------- Notification and integration ----------

    def test_notify_logs_and_sounds_once_for_a_batch(self):
        """Log every simultaneous entry but play a single alert sound."""
        self.alerts.sound = True
        entries = self.alerts.check_entries([self.person(1), self.person(2)], {1, 2})
        with patch.object(Utils, "play_alert_sound") as sound:
            with self.assertLogs(level="WARNING") as logs:
                self.alerts.notify(entries)
            self.alerts.notify([])
            sound.assert_called_once()
        self.assertEqual(len(logs.output), 2)
        self.assertIn("person #2", self.alerts.last_message)

    def test_pipeline_alerts_when_a_person_crosses_the_boundary(self):
        """Use actual polygon membership with the event service in the frame flow."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = MagicMock()
        tracker = MagicMock()
        tracker.get_remembered_ids.return_value = {1}
        with patch.object(self.alerts, "notify") as notify:
            for bottom in (90, 50, 50, 90, 50):
                tracker.update.return_value = [dict(self.person(), bbox=[40, 0, 60, bottom])]
                process_frame(frame, detector, tracker, [self.zone], 0, self.alerts)
            self.assertEqual([len(call.args[0]) for call in notify.call_args_list], [0, 1, 0, 0, 1])

    def test_alert_mode_enables_tracking_and_cleans_up(self):
        """The user needs only --alerts to run the full monitoring pipeline."""
        config = Utils.load_camera_config("config/camera.example.json")
        config["zones"] = [self.zone]
        config["alerts"] = {"sound": False}
        with patch("services.video_service.Camera") as camera_class, \
                patch("services.video_service.Detector") as detector_class, \
                patch("services.video_service.Tracker") as tracker_class:
            camera = camera_class.return_value
            camera.get_fps.return_value = 25
            camera.read_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            tracker_class.return_value.update.return_value = []
            self.assertEqual(run_video(config, headless=True, max_frames=1, alerts=True), 1)
            detector_class.assert_called_once_with(config)
            tracker_class.assert_called_once_with(config, frame_rate=25)
            camera.close.assert_called_once()

    def test_alert_mode_requires_a_restricted_zone(self):
        """Fail clearly instead of silently running without any alert area."""
        with self.assertRaises(ValueError):
            EntryAlerts(self.config, [])


if __name__ == "__main__":
    unittest.main()
