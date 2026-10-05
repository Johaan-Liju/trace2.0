"""Phase 2: turn original camera frames into simple YOLO detections."""

import logging

from vision.utils import TraceUtils


class Detector:
    """Load one pretrained YOLO model and reuse it for every frame."""

    # ---------- Model setup ----------

    def __init__(self, config):
        """Validate settings and load the model once, before reading video."""
        self.settings = TraceUtils.get_detection_config(config)
        self.model = self.load_model()
        if self.model.task != "detect":
            raise ValueError("Choose an object-detection model, such as yolov8n.pt.")
        for class_id in self.settings["classes"]:
            if class_id not in self.model.names:
                raise ValueError(f"Class ID {class_id} is not available in this model.")

    def load_model(self):
        """Import YOLO only when needed; first use may download its weights."""
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO dependencies are missing. Run: python -m pip install -r requirements.txt"
            ) from error

        logging.info("Loading YOLO. First use may download pretrained weights.")
        try:
            return YOLO(self.settings["model"])
        except Exception as error:
            raise RuntimeError(
                "Cannot load the YOLO model. Check its path, download connection, "
                "and installed dependencies."
            ) from error

    # ---------- Frame detection ----------

    def detect(self, frame, confidence=None):
        """Return class, confidence, and [x1, y1, x2, y2] for each detection."""
        # Tracking keeps weaker detections to help match existing people.
        if confidence is None:
            confidence = self.settings["confidence"]
        result = self.model.predict(
            source=frame,
            conf=confidence,
            imgsz=self.settings["image_size"],
            device=self.settings["device"],
            classes=self.settings["classes"],
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        # Ultralytics returns one row per box: x1, y1, x2, y2, score, class ID.
        for row in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, confidence, class_id = row
            detections.append({
                "class": result.names[int(class_id)],
                "class_id": int(class_id),
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        return detections
