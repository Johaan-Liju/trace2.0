"""Detection checks without downloading weights or requiring a GPU."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.video_service import Detector
from services.utils import Utils
from services.video_service import run_video


class DetectionTests(unittest.TestCase):
    """Check the model boundary, preview coordinates, and worker cleanup."""

    # ---------- Model output checks ----------

    def test_detection_output_and_model_settings(self):
        """Convert model tensors into plain Python values at original scale."""
        model = MagicMock()
        model.task = "detect"
        model.names = {0: "person"}
        result = model.predict.return_value[0]
        result.names = model.names
        result.boxes.data.cpu.return_value.tolist.return_value = [
            [10.5, 20, 100, 110, 0.91, 0]
        ]
        with patch.object(Detector, "load_model", return_value=model):
            detector = Detector({})
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [{
            "class": "person", "class_id": 0, "confidence": 0.91,
            "bbox": [10.5, 20.0, 100.0, 110.0],
        }])
        options = model.predict.call_args.kwargs
        self.assertIs(options["source"], frame)
        self.assertEqual(options["classes"], [0])
        self.assertEqual(options["device"], "cpu")
        result.boxes.data.cpu.return_value.tolist.return_value = []
        self.assertEqual(detector.detect(frame), [])

    def test_invalid_detection_settings(self):
        """Reject bad settings before loading a potentially large model."""
        invalid = [None, {"confidence": float("nan")}, {"confidence": True},
                   {"image_size": 641}, {"classes": []}, {"classes": [True]},
                   {"device": ""}, {"model": ""}]
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    Utils.get_detection_config({"detection": settings})

    # ---------- Preview and integration checks ----------

    def test_drawing_preserves_original_frame_and_box_scale(self):
        """Only the display copy gets boxes; resizing keeps them aligned."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        detections = [{"class": "person", "confidence": 0.9,
                       "bbox": [40, 40, 100, 100]}]
        annotated = Utils.draw_detections(frame, detections)
        preview = Utils.resize_frame(annotated, 320)
        self.assertFalse(frame.any())
        self.assertGreater(preview[140, 80, 1], 0)

    def test_worker_detects_before_resizing_and_releases_on_failure(self):
        """Pass original frames to detection and close on inference errors."""
        config = Utils.load_camera_config("config/camera.example.json")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        with patch("services.video_service.Camera") as camera_class:
            with patch("services.video_service.Detector") as detector_class:
                camera = camera_class.return_value
                camera.get_fps.return_value = 25
                camera.read_frame.return_value = frame
                detector = detector_class.return_value
                detector.detect.return_value = []
                self.assertEqual(run_video(config, True, 1, detect=True), 1)
                self.assertIs(detector.detect.call_args.args[0], frame)
                detector.detect.side_effect = RuntimeError("Inference failed")
                with self.assertRaises(RuntimeError):
                    run_video(config, True, 1, detect=True)
                self.assertEqual(camera.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
