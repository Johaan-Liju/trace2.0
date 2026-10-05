"""Check polygon membership, ignore precedence, and zone-editor saves."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from services.utils import Utils
from services.zone_service import ZoneEditor
from services.video_service import run_video


class ZoneTests(unittest.TestCase):
    """Use synthetic geometry so zone correctness does not depend on YOLO."""

    # ---------- Test fixtures ----------

    def zone(self, zone_id="restricted_01", zone_type="restricted", points=None):
        """Build a rectangle occupying the center of the image."""
        return {"id": zone_id, "name": zone_id, "type": zone_type,
                "points": points if points is not None else
                [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}

    def person(self, bbox):
        """Build a tracked person for membership checks."""
        return {"track_id": 1, "class": "person", "confidence": 0.9, "bbox": bbox}

    # ---------- Polygon membership ----------

    def test_inside_outside_and_boundary_use_feet(self):
        """Shoulder overlap alone does not put someone's feet in the zone."""
        zones = Utils.get_zones({"zones": [self.zone()]})
        people = [self.person([40, 0, 60, 50]), self.person([40, 30, 60, 90]),
                  self.person([15, 0, 35, 25])]
        assigned = Utils.assign_zones(people, zones, (100, 100, 3))
        self.assertEqual([person["current_zone"] for person in assigned],
                         ["restricted_01", None, "restricted_01"])
        self.assertNotIn("zone_ids", people[0])
        self.assertEqual(Utils.count_zone_occupants(assigned, zones), {"restricted_01": 2})

    def test_zone_scales_with_frame_resolution(self):
        """Changing image size preserves relative polygon membership."""
        zones = Utils.get_zones({"zones": [self.zone()]})
        for size in (100, 200, 800):
            track = self.person([size * 0.4, 0, size * 0.6, size * 0.5])
            assigned = Utils.assign_zones([track], zones, (size, size, 3))
            self.assertEqual(assigned[0]["current_zone"], "restricted_01")

    def test_concave_polygon(self):
        """Accept an L-shaped zone without counting its missing corner."""
        points = [[0, 0], [1, 0], [1, 0.3], [0.3, 0.3], [0.3, 1], [0, 1]]
        zones = Utils.get_zones({"zones": [self.zone(points=points)]})
        assigned = Utils.assign_zones([self.person([0, 0, 20, 80]),
                                      self.person([60, 0, 80, 80])], zones, (100, 100, 3))
        self.assertIsNotNone(assigned[0]["current_zone"])
        self.assertIsNone(assigned[1]["current_zone"])

    def test_overlapping_zones_and_ignore_precedence(self):
        """Preserve all active memberships until an ignore polygon overrides them."""
        zones = [self.zone(), self.zone("waiting", "loitering"), self.zone("crowd", "crowd")]
        person = self.person([40, 0, 60, 50])
        zones = Utils.get_zones({"zones": zones})
        assigned = Utils.assign_zones([person], zones, (100, 100, 3))
        self.assertEqual(assigned[0]["zone_ids"], ["restricted_01", "waiting", "crowd"])
        zones = Utils.get_zones({"zones": zones + [self.zone("ignored", "ignore")]})
        assigned = Utils.assign_zones([person], zones, (100, 100, 3))
        self.assertTrue(assigned[0]["ignored"])
        self.assertEqual(assigned[0]["zone_ids"], ["ignored"])
        self.assertEqual(Utils.count_zone_occupants(assigned, zones),
                         {"restricted_01": 0, "waiting": 0, "crowd": 0, "ignored": 1})

    def test_invalid_zones_are_rejected(self):
        """Reject malformed points, self-crossing polygons, and duplicate IDs."""
        invalid = [None, [], [[0, 0], [1, 1]], [[0, 0], [0.5, 0.5], [1, 1]],
                   [[0, 0], [1, 0], [2, 1]], [[0, 0], [1, 0], [0, float("nan")]],
                   [[0, 0], [1, 0], [1, 1], [0, 0]],
                   [[0, 0], [1, 1], [0, 1], [1, 0.2]],
                   [[0, 0], [1, 0], [0.5, 0], [1, 1], [0, 1]],
                   [[False, 0], [1, 0], [1, 1]]]
        for points in invalid:
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    Utils.get_zones({"zones": [dict(self.zone(), points=points)]})
        for zones in ({}, [self.zone(), self.zone()], [self.zone(zone_type="unknown")]):
            with self.assertRaises(ValueError):
                Utils.get_zones({"zones": zones})

    def test_draw_zones_preserves_original_frame(self):
        """A saved preview contains zone outlines without changing model input."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        preview = Utils.draw_zones(frame, [self.zone()])
        self.assertFalse(frame.any())
        self.assertTrue(preview[100, 50].any())

    def test_worker_includes_membership_in_drawn_tracks(self):
        """The monitoring pipeline sends enriched tracks to the shared renderer."""
        config = Utils.load_camera_config("config/camera.example.json")
        config["zones"] = [self.zone()]
        with patch("services.video_service.Camera") as camera_class, \
                patch("services.video_service.Detector"), patch("services.video_service.Tracker") as tracker_class, \
                patch("services.video_service.Utils.draw_detections", side_effect=lambda frame, tracks: frame) as draw:
            camera = camera_class.return_value
            camera.get_fps.return_value = 25
            camera.read_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            tracker_class.return_value.update.return_value = [self.person([40, 0, 60, 50])]
            self.assertEqual(run_video(config, True, 1, track=True), 1)
            self.assertEqual(draw.call_args.args[1][0]["current_zone"], "restricted_01")


class ZoneEditorTests(unittest.TestCase):
    """Check the editor without opening a desktop window or camera."""

    # ---------- Editor setup ----------

    def setUp(self):
        """Create a temporary camera config with an existing unrelated zone."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "camera.json"
        self.config = {"camera_id": "camera_01", "name": "Test", "source": "original.mp4",
                       "tracking": {"track_buffer_seconds": 8}, "zones": [
                           {"id": "other", "name": "Other", "type": "crowd",
                            "points": [[0, 0], [0.2, 0], [0.2, 0.2]]}]}
        self.path.write_text(json.dumps(self.config), encoding="utf-8")

    # ---------- Input and persistence ----------

    def test_click_mapping_and_save_preserve_camera_settings(self):
        """Normalize preview clicks and avoid persisting environment credentials."""
        with patch.dict("os.environ", {"TRACE_CAMERA_SOURCE": "rtsp://private/override"}):
            editor = ZoneEditor(self.path, "new", "Door", "restricted")
        editor.preview = np.zeros((100, 200, 3), dtype=np.uint8)
        for x, y in [(50, 25), (150, 25), (150, 75), (50, 75)]:
            editor.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, None, None)
        editor.save_zone()
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["source"], "original.mp4")
        self.assertEqual(saved["tracking"], self.config["tracking"])
        self.assertEqual(saved["zones"][0], self.config["zones"][0])
        self.assertEqual(saved["zones"][1]["points"][0], [0.25, 0.25])
        edited = ZoneEditor(self.path, "new")
        self.assertEqual(edited.zone["name"], "Door")
        edited.save_zone()
        self.assertEqual(len(json.loads(self.path.read_text())["zones"]), 2)

    def test_invalid_polygon_does_not_change_config(self):
        """Saving an unfinished polygon leaves the original file untouched."""
        before = self.path.read_bytes()
        editor = ZoneEditor(self.path, "new")
        editor.points = [[0, 0], [1, 1]]
        with self.assertRaises(ValueError):
            editor.save_zone()
        self.assertEqual(self.path.read_bytes(), before)

    def test_clicking_first_corner_does_not_add_duplicate(self):
        """Allow the natural close-polygon click before saving."""
        editor = ZoneEditor(self.path, "new")
        editor.preview = np.zeros((100, 200, 3), dtype=np.uint8)
        for x, y in [(20, 20), (180, 20), (180, 80), (20, 80), (20, 20)]:
            editor.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, None, None)
        self.assertEqual(len(editor.points), 4)
        self.assertTrue(editor.handle_action("save"))

    def test_save_button_click_does_not_become_a_polygon_point(self):
        """The toolbar works with a mouse and does not change polygon geometry."""
        editor = ZoneEditor(self.path, "new")
        editor.preview = np.zeros((100, 200, 3), dtype=np.uint8)
        editor.points = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        action, (left, top, right, bottom) = editor.get_buttons()[0]
        editor.on_mouse(cv2.EVENT_LBUTTONDOWN, (left + right) // 2, (top + bottom) // 2, None, None)
        self.assertEqual(editor.pending_action, "save")
        self.assertEqual(len(editor.points), 4)
        self.assertTrue(editor.handle_action(editor.pending_action))

    def test_tall_frame_fits_with_room_for_controls(self):
        """A portrait source does not push the Save button below the screen."""
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        preview = Utils.fit_frame(frame, 960, 540)
        self.assertLessEqual(preview.shape[0], 540)
        self.assertLessEqual(preview.shape[1], 960)
        self.assertAlmostEqual(preview.shape[1] / preview.shape[0], 1080 / 1920, places=2)

    def test_cancel_after_invalid_save_keeps_original_config(self):
        """Keep the editor open after invalid Enter, then cancel without writing."""
        before = self.path.read_bytes()
        editor = ZoneEditor(self.path, "new")
        editor.preview = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch.object(editor, "capture_preview"), \
                patch("services.zone_service.cv2.namedWindow"), \
                patch("services.zone_service.cv2.setMouseCallback"), \
                patch("services.zone_service.cv2.imshow"), \
                patch("services.zone_service.cv2.getWindowProperty", return_value=1), \
                patch("services.zone_service.cv2.waitKey", side_effect=[13, 27]), \
                patch("services.zone_service.cv2.destroyAllWindows") as close:
            self.assertFalse(editor.run())
            close.assert_called_once()
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
