"""Shared helpers. Keep repeated configuration and frame operations here."""

import json
import logging
import math
import os
import tempfile
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


class Utils:
    """Small, reusable functions that do not keep camera state."""

    ZONE_TYPES = ("restricted", "loitering", "crowd", "ignore")

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

    # ---------- Clip settings and encoding ----------

    @staticmethod
    def get_clip_config(config):
        """Read optional clip settings and locate the installed FFmpeg program."""
        settings = {"output_dir": "storage/clips", "duration_seconds": 5, "fps": 15,
                    "reentry_gap_seconds": 2}
        supplied = config.get("clips", {})
        if not isinstance(supplied, dict):
            raise ValueError("'clips' must be a JSON object.")
        settings.update(supplied)
        for field, maximum in (("duration_seconds", 60), ("fps", 60), ("reentry_gap_seconds", 60)):
            value = settings[field]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"Clips '{field}' must be greater than 0 and at most {maximum}.")
        if not isinstance(settings["output_dir"], str) or not settings["output_dir"].strip():
            raise ValueError("Clips 'output_dir' must be a non-empty folder path.")
        settings["ffmpeg"] = shutil.which("ffmpeg")
        if settings["ffmpeg"] is None:
            raise RuntimeError("FFmpeg is missing. Install FFmpeg and add it to PATH before using --clips.")
        return settings

    @staticmethod
    def encode_clip(folder, frames, end_time, output_path, settings):
        """Encode timestamped JPEG frames as one silent MP4; return its path."""
        # FFmpeg's concat input preserves timing even when live detection is slow.
        lines = []
        for index, (name, timestamp) in enumerate(frames):
            next_time = frames[index + 1][1] if index + 1 < len(frames) else end_time
            lines.extend([f"file '{name}'", f"duration {next_time - timestamp:.9f}"])
        lines.append(f"file '{frames[-1][0]}'")
        manifest = Path(folder) / "frames.txt"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        partial_path = output_path.with_suffix(".partial.mp4")
        command = [
            settings["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "concat", "-safe", "1", "-i", str(manifest),
            "-t", str(end_time - frames[0][1]), "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", str(settings["fps"]),
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(partial_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=60,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode != 0 or not partial_path.is_file() or partial_path.stat().st_size == 0:
                raise RuntimeError("FFmpeg could not encode the entry clip. Check disk space and H.264 support.")
            partial_path.rename(output_path)
            return output_path
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("FFmpeg took too long to finish the entry clip.") from error
        finally:
            partial_path.unlink(missing_ok=True)

    # ---------- Local notification ----------

    @staticmethod
    def play_alert_sound():
        """Play the Windows alert sound asynchronously without stopping capture."""
        try:
            import winsound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except (ImportError, RuntimeError):
            logging.warning("Alert sound unavailable; the on-screen and terminal alerts remain active.")

    @staticmethod
    def draw_alert_banner(frame, message):
        """Show the latest entry below the camera timestamp in the preview."""
        preview = frame.copy()
        Utils.draw_text_label(preview, message, (12, 52), (0, 80, 255))
        return preview

    # ---------- Frame display ----------

    @staticmethod
    def build_preview(frame, config, frame_number, alert_message=""):
        """Resize a processed frame and add the camera label and latest alert."""
        preview = Utils.resize_frame(frame, config["display_width"])
        preview = Utils.add_preview_label(preview, config["camera_id"], frame_number)
        if alert_message:
            preview = Utils.draw_alert_banner(preview, alert_message)
        return preview

    @staticmethod
    def show_preview(window_name, preview):
        """Display a frame; return False when Q is pressed or the window closes."""
        cv2.imshow(window_name, preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1

    @staticmethod
    def draw_frame_results(frame, detections, zones, detection_enabled):
        """Draw zones first, then detected boxes; plain camera mode omits counts."""
        annotated = frame
        if zones:
            counts = None
            if detection_enabled:
                counts = Utils.count_zone_occupants(detections, zones)
            annotated = Utils.draw_zones(annotated, zones, counts)
        if detection_enabled:
            annotated = Utils.draw_detections(annotated, detections)
        return annotated

    # ---------- Zone and detection drawing ----------

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
            polygon = np.rint(Utils.zone_contour(zone, frame.shape)).astype(np.int32)
            color = Utils.get_zone_color(zone["type"])
            cv2.polylines(annotated, [polygon], True, color, 2)
            x, y = polygon[0]
            x = max(0, min(int(x), frame.shape[1] - 1))
            y = max(45, min(int(y) - 8, frame.shape[0] - 10))
            label = f"{zone['name']} ({zone['type']})"
            if counts is not None:
                label += f": {counts[zone['id']]}"
            Utils.draw_text_label(annotated, label, (x, y), color)
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
                    color = Utils.get_zone_color("ignore")
                    zone_label += " [ignored]"
                elif "restricted" in detection["zone_types"]:
                    color = Utils.get_zone_color("restricted")
                foot = tuple(round(value) for value in Utils.get_foot_point(detection["bbox"]))
                cv2.circle(annotated, foot, 5, color, -1)
                Utils.draw_text_label(annotated, zone_label, (x1, y1 + 18), color, scale=0.45)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            Utils.draw_text_label(annotated, label, (x1, y1 - 8), color)
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
                if Utils.segments_intersect(points[first], points[(first + 1) % count],
                                                points[second], points[(second + 1) % count]):
                    raise ValueError("Polygon edges must not cross or touch other edges.")

    # ---------- Zone membership ----------

    @staticmethod
    def get_zones(config):
        """Load this camera's zones and reject invalid or duplicate polygons."""
        zones = config.get("zones", [])
        if not isinstance(zones, list):
            raise ValueError("'zones' must be a list.")

        used_ids = set()
        for zone in zones:
            if not isinstance(zone, dict):
                raise ValueError("Each zone must be a JSON object.")
            for field in ("id", "name", "type"):
                if not isinstance(zone.get(field), str) or not zone[field].strip():
                    raise ValueError(f"Zone '{field}' must be a non-empty string.")
            if zone["id"] in used_ids:
                raise ValueError(f"Duplicate zone ID: {zone['id']}")
            used_ids.add(zone["id"])
            if zone["type"] not in Utils.ZONE_TYPES:
                raise ValueError(f"Zone type must be one of: {', '.join(Utils.ZONE_TYPES)}.")
            Utils.validate_polygon(zone.get("points"))
        return zones

    @staticmethod
    def assign_zones(tracks, zones, frame_shape):
        """Return copies of tracks with their current zone memberships."""
        polygons = [(zone, Utils.zone_contour(zone, frame_shape)) for zone in zones]
        assigned_tracks = []

        for track in tracks:
            point = Utils.get_foot_point(track["bbox"])
            matches = []
            for zone, polygon in polygons:
                # A point on an edge or vertex counts as inside.
                if cv2.pointPolygonTest(polygon, point, False) >= 0:
                    matches.append(zone)

            ignored_zones = [zone for zone in matches if zone["type"] == "ignore"]
            if ignored_zones:
                matches = ignored_zones

            assigned = dict(track)
            assigned["zone_ids"] = [zone["id"] for zone in matches]
            assigned["zone_names"] = [zone["name"] for zone in matches]
            assigned["zone_types"] = [zone["type"] for zone in matches]
            assigned["current_zone"] = matches[0]["id"] if matches else None
            assigned["ignored"] = bool(ignored_zones)
            assigned_tracks.append(assigned)

        return assigned_tracks

    @staticmethod
    def count_zone_occupants(tracks, zones):
        """Count current occupants; ignore zones override overlapping active zones."""
        counts = {zone["id"]: 0 for zone in zones}
        for track in tracks:
            for zone_id in track["zone_ids"]:
                counts[zone_id] += 1
        return counts
