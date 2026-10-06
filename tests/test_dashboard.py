"""Exercise dashboard lifecycle, actual HTTP routes, and config boundaries."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from services.web_service import Dashboard, create_server


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "camera.json"
        self.path.write_text(json.dumps({"camera_id": "test", "name": "Test camera", "source": "0",
                                         "display_width": 160, "alerts": {"sound": False},
                                         "zones": []}), encoding="utf-8")
        self.dashboard = Dashboard(self.path)

    def test_invalid_save_is_atomic_and_private_override_is_not_persisted(self):
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.dashboard.save({"name": "Changed", "zones": [{"id": "bad"}]})
        self.assertEqual(self.path.read_bytes(), before)
        with patch.dict("os.environ", {"TRACE_CAMERA_SOURCE": "rtsp://user:secret@camera/live"}):
            dashboard = Dashboard(self.path)
            self.assertEqual(dashboard.snapshot()["config"]["source"], "")
            dashboard.save({"name": "New camera", "confidence": 0.65})
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["source"], "0")
        self.assertEqual(saved["tracking"]["new_track_thresh"], 0.65)
        self.assertEqual(saved["tracking"]["track_high_thresh"], 0.65)

    def test_monitoring_collects_entries_frames_and_final_clip(self):
        zone = {"id": "z", "name": "Entrance", "type": "restricted",
                "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        self.dashboard.save({"zones": [zone], "clips": True})
        track = {"track_id": 7, "class": "person", "class_id": 0,
                 "confidence": .9, "bbox": [10, 10, 30, 50]}
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        with patch("services.web_service.Camera") as camera_class, \
             patch("services.web_service.Detector"), patch("services.web_service.Tracker") as tracker_class, \
             patch("services.web_service.ClipRecorder") as recorder_class, \
             patch("services.web_service.Utils.get_clip_config"):
            camera = camera_class.return_value
            camera.source_type = "file"
            camera.get_fps.return_value = 1000
            camera.read_frame.side_effect = [frame, frame, None]
            tracker_class.return_value.update.return_value = [track]
            tracker_class.return_value.get_remembered_ids.return_value = {7}
            recorder_class.return_value.update.return_value = None
            recorder_class.return_value.close.return_value = Path(self.folder.name) / "finished.mp4"
            self.dashboard.start()
            self.dashboard.worker.join(5)
            self.assertFalse(self.dashboard.worker.is_alive())
            snapshot = self.dashboard.snapshot()
            self.assertEqual(snapshot["status"], "ended")
            self.assertEqual(snapshot["frames"], 2)
            self.assertEqual(snapshot["total_alerts"], 1)
            self.assertEqual(snapshot["events"][0]["track_id"], 7)
            self.assertEqual(len(snapshot["clips"]), 1)
            self.assertTrue(snapshot["has_frame"])
            camera.close.assert_called_once()
            recorder_class.return_value.close.assert_called_once()

    def test_start_stop_rejects_duplicate_workers_and_releases_camera(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def read():
            entered.set()
            release.wait(5)
            return np.zeros((90, 160, 3), dtype=np.uint8)

        with patch("services.web_service.Camera") as camera_class, patch("services.web_service.Detector") as detector:
            camera = camera_class.return_value
            camera.read_frame.side_effect = read
            camera.get_fps.return_value = 25
            self.dashboard.start("preview")
            self.assertTrue(entered.wait(3))
            with self.assertRaises(ValueError):
                self.dashboard.start()
            with self.assertRaises(ValueError):
                self.dashboard.save({"name": "Busy"})
            self.assertEqual(self.dashboard.stop()["status"], "stopping")
            release.set()
            self.dashboard.worker.join(5)
            self.assertFalse(self.dashboard.snapshot()["busy"])
            self.assertEqual(self.dashboard.status, "idle")
            camera.close.assert_called_once()
            detector.assert_not_called()

    def test_failed_worker_reports_error_and_releases_camera(self):
        with patch("services.web_service.Camera") as camera_class, patch("services.web_service.Detector") as detector:
            detector.side_effect = RuntimeError("rtsp://user:secret@private-camera/live")
            self.dashboard.start()
            self.dashboard.worker.join(5)
            snapshot = self.dashboard.snapshot()
            self.assertEqual(snapshot["status"], "error")
            self.assertNotIn("secret", json.dumps(snapshot))
            camera_class.return_value.close.assert_called_once()

    def test_http_preview_security_and_clip_download(self):
        video = Path(self.folder.name) / "input.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 90))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.full((90, 160, 3), 100, np.uint8))
        writer.release()
        self.dashboard.save({"source": str(video)})
        server = create_server(self.dashboard, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, payload=None, headers=None):
            data = json.dumps(payload).encode() if payload is not None else None
            return urlopen(Request(base + path, data=data, headers=headers or {}), timeout=5)

        with request("/") as response:
            self.assertIn(b"Camera intelligence", response.read())
        with request("/api/state") as response:
            token = json.load(response)["token"]
        for path, payload, headers, code in [
            ("/api/start", {}, {}, 403),
            ("/api/start", {}, {"X-Trace-Token": token, "Origin": "https://example.com"}, 403),
            ("/api/state", None, {"Host": "example.com"}, 403),
            ("/../config/camera.local.json", None, {}, 404),
            ("/clips/../../config/camera.local.json", None, {}, 404),
            ("/api/config", {"zones": [{"id": "bad"}]}, {"X-Trace-Token": token}, 400),
        ]:
            with self.subTest(path=path, headers=headers):
                with self.assertRaises(HTTPError) as caught:
                    request(path, payload, headers)
                self.assertEqual(caught.exception.code, code)
                caught.exception.close()
        with request("/api/start", {"mode": "preview"}, {"X-Trace-Token": token}) as response:
            self.assertEqual(response.status, 200)
        self.dashboard.worker.join(5)
        with request("/api/frame.jpg") as response:
            frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(frame.shape, (90, 160, 3))
        self.assertEqual(self.dashboard.status, "ended")
        clip = Path(self.folder.name) / "clip.mp4"
        clip.write_bytes(b"test clip download")
        self.dashboard._add_clip(clip)
        with request(self.dashboard.snapshot()["clips"][0]["url"]) as response:
            self.assertEqual(response.read(), b"test clip download")
            self.assertIn("attachment", response.headers["Content-Disposition"])


if __name__ == "__main__":
    unittest.main()
