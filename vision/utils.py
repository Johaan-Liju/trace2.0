"""Shared helpers. Keep repeated configuration and frame operations here."""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


class TraceUtils:
    """Small, reusable functions that do not keep camera state."""

    # ---------- Configuration helpers ----------

    @staticmethod
    def save_json(path, data):
        """Write JSON to a temporary file, then replace the original safely."""
        path = Path(path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(data, temporary, indent=2, allow_nan=False)
                temporary.write("\n")
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def load_camera_config(config_path):
        """Read JSON, apply an optional private source, and validate settings."""
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)

        if not isinstance(config, dict):
            raise ValueError("Camera configuration must be a JSON object.")

        for field in ("camera_id", "name", "source"):
            if field == "source" and os.environ.get("TRACE_CAMERA_SOURCE"):
                config[field] = os.environ["TRACE_CAMERA_SOURCE"]
            value = config.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{field}' must be a non-empty string.")
            config[field] = value.strip()

        defaults = {
            "display_width": 960,
            "fallback_fps": 25,
            "reconnect_attempts": 3,
            "reconnect_delay_seconds": 2,
            "timeout_ms": 5000,
        }
        for field, default in defaults.items():
            value = config.setdefault(field, default)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"'{field}' must be a finite number.")
            minimum = 0 if field == "reconnect_attempts" else 0.001
            if value < minimum:
                raise ValueError(f"'{field}' must be at least {minimum}.")

        for field in ("display_width", "reconnect_attempts", "timeout_ms"):
            if type(config[field]) is not int:
                raise ValueError(f"'{field}' must be a whole number.")

        return config

    @staticmethod
    def get_detection_config(config):
        """Read optional YOLO settings; older camera config files still work."""
        settings = {
            "model": "yolov8n.pt",
            "confidence": 0.4,
            "image_size": 640,
            "device": "cpu",
            "classes": [0],
        }
        supplied = config.get("detection", {})
        if not isinstance(supplied, dict):
            raise ValueError("'detection' must be a JSON object.")
        settings.update(supplied)

        for field in ("model", "device"):
            if not isinstance(settings[field], str) or not settings[field].strip():
                raise ValueError(f"Detection '{field}' must be a non-empty string.")

        confidence = settings["confidence"]
        if type(confidence) not in (int, float) or not 0 < confidence <= 1:
            raise ValueError("Detection 'confidence' must be greater than 0 and at most 1.")
        size = settings["image_size"]
        if type(size) is not int or size < 32 or size % 32 != 0:
            raise ValueError("Detection 'image_size' must be a positive multiple of 32.")
        classes = settings["classes"]
        if not isinstance(classes, list) or not classes:
            raise ValueError("Detection 'classes' must be a non-empty list of class IDs.")
        if any(type(class_id) is not int or class_id < 0 for class_id in classes):
            raise ValueError("Each detection class ID must be a non-negative integer.")
        return settings

    @staticmethod
    def get_tracking_config(config, frame_rate=30):
        """Convert the requested gap duration to ByteTrack's frame buffer."""
        settings = {
            "track_high_thresh": 0.4,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.4,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True,
        }
        supplied = config.get("tracking", {})
        if not isinstance(supplied, dict):
            raise ValueError("'tracking' must be a JSON object.")
        settings.update(supplied)

        # Keep older configs with an explicit frame buffer working as before.
        # New configs use five seconds of source frames by default.
        if "track_buffer_seconds" in supplied or "track_buffer" not in supplied:
            seconds = supplied.get("track_buffer_seconds", 5)
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Tracking 'track_buffer_seconds' must be a positive finite number.")
            if type(frame_rate) not in (int, float) or not math.isfinite(frame_rate) or frame_rate <= 0:
                raise ValueError("Tracking frame rate must be a positive finite number.")
            settings["track_buffer"] = max(1, round(seconds * frame_rate))

        for field in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            value = settings[field]
            if type(value) not in (int, float) or not 0 < value <= 1:
                raise ValueError(f"Tracking '{field}' must be greater than 0 and at most 1.")
        if not settings["track_low_thresh"] < settings["track_high_thresh"] <= settings["new_track_thresh"]:
            raise ValueError("Tracking thresholds must satisfy: low < high <= new.")
        if type(settings["track_buffer"]) is not int or settings["track_buffer"] < 1:
            raise ValueError("Tracking 'track_buffer' must be a positive whole number.")
        if type(settings["fuse_score"]) is not bool:
            raise ValueError("Tracking 'fuse_score' must be true or false.")
        return settings

    @staticmethod
    def get_source_type(source):
        """Identify a webcam index, an RTSP stream, or a local video file."""
        if source.isdigit():
            return "webcam"
        if source.lower().startswith(("rtsp://", "rtsps://")):
            return "rtsp"
        return "file"

    # ---------- Frame helpers ----------

    @staticmethod
    def get_foot_point(bbox):
        """Use the bottom-center of a box as a person's position on the floor."""
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2), float(y2))

    @staticmethod
    def zone_contour(zone, frame_shape):
        """Map normalized zone points to the original image's pixel coordinates."""
        height, width = frame_shape[:2]
        return np.asarray([[x * width, y * height] for x, y in zone["points"]], dtype=np.float32)

    @staticmethod
    def get_zone_color(zone_type):
        """Return consistent OpenCV BGR colors for zone outlines and labels."""
        colors = {"restricted": (0, 80, 255), "loitering": (0, 210, 255),
                  "crowd": (255, 180, 0), "ignore": (160, 160, 160)}
        return colors[zone_type]

    @staticmethod
    def draw_text_label(frame, text, position, color, scale=0.5):
        """Draw readable text on a preview copy, keeping it within the frame."""
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max(0, min(int(position[0]), frame.shape[1] - width - 6))
        y = max(height + 3, min(int(position[1]), frame.shape[0] - baseline - 3))
        cv2.rectangle(frame, (x, y - height - 3), (x + width + 6, y + baseline + 3), (25, 25, 25), -1)
        cv2.putText(frame, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_zones(frame, zones, counts=None):
        """Draw zone boundaries and optional occupant counts on a copy."""
        annotated = frame.copy()
        for zone in zones:
            polygon = np.rint(TraceUtils.zone_contour(zone, frame.shape)).astype(np.int32)
            color = TraceUtils.get_zone_color(zone["type"])
            cv2.polylines(annotated, [polygon], True, color, 2)
            x, y = polygon[0]
            x = max(0, min(int(x), frame.shape[1] - 1))
            y = max(45, min(int(y) - 8, frame.shape[0] - 10))
            label = f"{zone['name']} ({zone['type']})"
            if counts is not None:
                label += f": {counts[zone['id']]}"
            TraceUtils.draw_text_label(annotated, label, (x, y), color)
        return annotated

    @staticmethod
    def draw_detections(frame, detections):
        """Draw detections or tracks on a copy, including IDs when available."""
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [round(value) for value in detection["bbox"]]
            name = detection["class"]
            if "track_id" in detection:
                name = f"{name} #{detection['track_id']}"
            label = f"{name} {detection['confidence']:.0%}"
            color = (0, 255, 0)
            if "zone_names" in detection:
                zone_label = ", ".join(detection["zone_names"]) or "Outside zones"
                if detection["ignored"]:
                    color = TraceUtils.get_zone_color("ignore")
                    zone_label += " [ignored]"
                elif "restricted" in detection["zone_types"]:
                    color = TraceUtils.get_zone_color("restricted")
                foot = tuple(round(value) for value in TraceUtils.get_foot_point(detection["bbox"]))
                cv2.circle(annotated, foot, 5, color, -1)
                TraceUtils.draw_text_label(annotated, zone_label, (x1, y1 + 18), color, scale=0.45)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            TraceUtils.draw_text_label(annotated, label, (x1, y1 - 8), color)
        return annotated

    @staticmethod
    def resize_frame(frame, width):
        """Resize a preview without stretching its original proportions."""
        original_height, original_width = frame.shape[:2]
        height = max(1, round(original_height * width / original_width))
        return cv2.resize(frame, (width, height))

    @staticmethod
    def fit_frame(frame, max_width, max_height):
        """Fit a frame within both limits so editor controls stay on screen."""
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))

    @staticmethod
    def add_preview_label(frame, camera_id, frame_number):
        """Label a preview copy, leaving the original frame ready for YOLO."""
        preview = frame.copy()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        text = f"{camera_id} | Frame {frame_number} | {timestamp}"
        cv2.putText(preview, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return preview

    @staticmethod
    def save_snapshot(frame, output_path):
        """Save one requested preview image; report a failed write clearly."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise OSError("Could not save the preview snapshot.")

    # ---------- Polygon validation ----------

    @staticmethod
    def segments_intersect(a, b, c, d):
        """Check whether two polygon edges cross or touch."""
        def cross(p, q, r):
            """Return which side of edge p-q contains point r."""
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
            return False
        if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
            return False
        return cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0

    @staticmethod
    def validate_polygon(points):
        """Require a non-crossing polygon with normalized coordinates from 0 to 1."""
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("A zone needs at least three corners.")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Each corner must be [x, y].")
            if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in point):
                raise ValueError("Zone coordinates must be numbers from 0 to 1.")
        if len({tuple(point) for point in points}) != len(points):
            raise ValueError("Do not repeat corners, including the first corner.")
        if cv2.contourArea(np.asarray(points, dtype=np.float32)) <= 0.000001:
            raise ValueError("The polygon must cover an area; its corners cannot form a line.")

        count = len(points)
        for first in range(count):
            a, b, c = points[first - 1], points[first], points[(first + 1) % count]
            incoming = (b[0] - a[0], b[1] - a[1])
            outgoing = (c[0] - b[0], c[1] - b[1])
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            if abs(cross) < 1e-12 and dot < 0:
                raise ValueError("Adjacent polygon edges must not fold back over each other.")
            for second in range(first + 1, count):
                # Adjacent edges share a corner and are allowed to meet there.
                if second == first + 1 or (first == 0 and second == count - 1):
                    continue
                if TraceUtils.segments_intersect(points[first], points[(first + 1) % count],
                                                points[second], points[(second + 1) % count]):
                    raise ValueError("Polygon edges must not cross or touch other edges.")
