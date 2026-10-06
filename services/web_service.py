"""Local HTTP dashboard and a single camera worker, sharing the existing model services."""

import copy
import json
import logging
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2

from services.clip_service import ClipRecorder
from services.event_service import EntryAlerts
from services.utils import Utils
from services.video_service import Camera, Detector, Tracker, detect_and_track


class Dashboard:
    """Own one monitoring session; HTTP threads only see locked snapshots."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = Utils.load_camera_config(self.config_path)
        Utils.get_zones(self.config)
        Utils.get_detection_config(self.config)
        Utils.get_tracking_config(self.config)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.token = secrets.token_urlsafe(32)
        self.status = "idle"
        self.error = ""
        self.jpeg = self.raw_jpeg = None
        self.frames = self.people = self.total_alerts = 0
        self.fps = 0
        self.events = deque(maxlen=500)
        self.clip_paths = {}
        self.zone_counts = {}

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def snapshot(self):
        with self.lock:
            config = self.config
            source_type = Utils.get_source_type(config["source"])
            settings = config.get("dashboard", {})
            return {
                "token": self.token, "status": self.status, "error": self.error,
                "busy": self.busy(), "has_frame": self.raw_jpeg is not None,
                "frames": self.frames, "people": self.people, "fps": round(self.fps, 1),
                "total_alerts": self.total_alerts, "events": list(reversed(self.events)),
                "clips": [{"id": key, "name": path.name, "url": f"/clips/{key}"}
                          for key, path in reversed(list(self.clip_paths.items()))],
                "zone_counts": dict(self.zone_counts),
                "config": {"name": config["name"], "camera_id": config["camera_id"],
                           "source": "" if source_type == "rtsp" else config["source"],
                           "source_type": source_type,
                           "confidence": Utils.get_tracking_config(config)["new_track_thresh"],
                           "model": Path(Utils.get_detection_config(config)["model"]).name,
                           "device": Utils.get_detection_config(config)["device"],
                           "zones": copy.deepcopy(Utils.get_zones(config)),
                           "alerts": settings.get("alerts", True),
                           "clips": settings.get("clips", False),
                           "sound": config.get("alerts", {}).get("sound", True)},
            }

    def save(self, payload):
        """Validate supported changes, preserve private source overrides, save atomically."""
        with self.lock:
            if self.busy():
                raise ValueError("Stop monitoring before changing camera settings or zones.")
            if not isinstance(payload, dict):
                raise ValueError("Settings must be a JSON object.")
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            for field in ("name", "source"):
                if field in payload:
                    value = payload[field]
                    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
                        raise ValueError(f"Enter a valid {field}.")
                    raw[field] = value.strip()
            if "confidence" in payload:
                raw.setdefault("detection", {})["confidence"] = payload["confidence"]
                raw.setdefault("tracking", {})["track_high_thresh"] = payload["confidence"]
                raw["tracking"]["new_track_thresh"] = payload["confidence"]
            if "zones" in payload:
                raw["zones"] = payload["zones"]
            for field in ("alerts", "clips", "sound"):
                if field in payload:
                    if type(payload[field]) is not bool:
                        raise ValueError(f"{field} must be true or false.")
                    if field == "sound":
                        raw.setdefault("alerts", {})["sound"] = payload[field]
                    else:
                        raw.setdefault("dashboard", {})[field] = payload[field]
            Utils.get_detection_config(raw)
            Utils.get_tracking_config(raw)
            Utils.get_zones(raw)
            Utils.save_json(self.config_path, raw)
            old_source = self.config["source"]
            self.config = Utils.load_camera_config(self.config_path)
            if old_source != self.config["source"]:
                self.jpeg = self.raw_jpeg = None
            self.zone_counts = {}
        return self.snapshot()

    def start(self, mode="monitor"):
        with self.lock:
            if self.busy():
                raise ValueError("The camera is already running.")
            if mode not in ("monitor", "preview"):
                raise ValueError("Choose monitor or preview mode.")
            config = copy.deepcopy(self.config)
            settings = config.get("dashboard", {})
            if mode == "monitor" and settings.get("clips", False):
                Utils.get_clip_config(config)
            self.stop_event.clear()
            self.status, self.error = "starting", ""
            self.frames = self.people = 0
            self.fps = 0
            self.zone_counts = {}
            self.jpeg = self.raw_jpeg = None
            self.worker = threading.Thread(target=self._run, args=(config, mode), daemon=True)
            self.worker.start()
        return self.snapshot()

    def stop(self):
        with self.lock:
            if self.busy():
                self.status = "stopping"
                self.stop_event.set()
        return self.snapshot()

    def _add_clip(self, path):
        if path:
            with self.lock:
                self.clip_paths[secrets.token_hex(12)] = Path(path)

    def _run(self, config, mode):
        camera, recorder = Camera(config), None
        final_status = "idle"
        try:
            zones = Utils.get_zones(config)
            settings = config.get("dashboard", {})
            monitor = mode == "monitor"
            alerts = EntryAlerts(config, zones) if monitor and settings.get("alerts", True) and any(
                zone["type"] == "restricted" for zone in zones) else None
            recorder = ClipRecorder(config) if monitor and settings.get("clips", False) else None
            detector = Detector(config) if monitor else None
            if self.stop_event.is_set():
                return
            if not camera.open():
                raise ConnectionError("Cannot open the camera. Check your source in Settings.")
            frame_rate = camera.get_fps()
            tracker = Tracker(config, frame_rate) if monitor else None
            connection = camera.connection_id
            frame_number = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                frame = camera.read_frame()
                if self.stop_event.is_set():
                    break
                if frame is None:
                    if frame_number == 0:
                        raise RuntimeError("No video frames could be decoded.")
                    final_status = "ended"
                    break
                if camera.connection_id != connection:
                    if tracker:
                        tracker.reset()
                    if alerts:
                        alerts.reset()
                    if recorder:
                        self._add_clip(recorder.reset())
                    connection = camera.connection_id
                timestamp = frame_number / frame_rate if camera.source_type == "file" else started
                tracks = detect_and_track(frame, detector, tracker, timestamp)
                tracks = Utils.assign_zones(tracks, zones, frame.shape)
                entries = alerts.check_entries(tracks, tracker.get_remembered_ids()) if alerts else []
                if alerts:
                    alerts.notify(entries)
                if recorder:
                    self._add_clip(recorder.update(frame, tracks, tracker.get_remembered_ids(), timestamp))
                annotated = Utils.draw_frame_results(frame, tracks, zones, monitor)
                preview = Utils.resize_frame(annotated, config["display_width"])
                raw = Utils.resize_frame(frame, config["display_width"])
                ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
                raw_ok, raw_jpeg = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok or not raw_ok:
                    raise RuntimeError("Could not encode the camera preview.")
                frame_number += 1
                if camera.source_type == "file":
                    self.stop_event.wait(max(0, 1 / frame_rate - (time.monotonic() - started)))
                with self.lock:
                    self.jpeg, self.raw_jpeg = jpeg.tobytes(), raw_jpeg.tobytes()
                    self.frames = frame_number
                    self.people = len(tracks)
                    self.fps = 1 / max(time.monotonic() - started, 0.001)
                    self.zone_counts = Utils.count_zone_occupants(tracks, zones)
                    self.events.extend(entries)
                    self.total_alerts += len(entries)
                    self.status = "stopping" if self.stop_event.is_set() else ("monitoring" if monitor else "preview")
        except Exception as error:
            # Native/model exceptions may contain a private RTSP source or credentials.
            logging.error("Dashboard camera worker failed (%s).", type(error).__name__)
            with self.lock:
                self.error = "Camera processing failed. Check the source, model, and settings, then try again."
            final_status = "error"
        finally:
            camera.close()
            try:
                if recorder:
                    self._add_clip(recorder.close())
            except Exception:
                with self.lock:
                    self.error = "The last clip could not be saved. Check FFmpeg and available disk space."
                final_status = "error"
            with self.lock:
                self.status = final_status
                self.people, self.fps = 0, 0
                self.zone_counts = {}


def create_server(dashboard, port=8765):
    """Serve only loopback requests, fixed assets, and explicitly registered clips."""
    assets = Path(__file__).resolve().parent.parent / "frontend"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _allowed(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (
                not origin or origin in {f"http://{host}" for host in hosts})

        def _send(self, body, content_type="application/json", status=200, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _json(self, data, status=200):
            self._send(json.dumps(data).encode(), status=status)

        def do_GET(self):
            if not self._allowed():
                self._json({"error": "Local dashboard requests only."}, 403)
                return
            path = urlsplit(self.path).path
            if path == "/api/state":
                self._json(dashboard.snapshot())
            elif path in ("/api/frame.jpg", "/api/raw.jpg"):
                with dashboard.lock:
                    frame = dashboard.raw_jpeg if path == "/api/raw.jpg" else dashboard.jpeg
                if frame:
                    self._send(frame, "image/jpeg")
                else:
                    self._json({"error": "No camera frame available yet."}, 404)
            elif path.startswith("/clips/"):
                with dashboard.lock:
                    clip = dashboard.clip_paths.get(path.removeprefix("/clips/"))
                if clip and clip.is_file():
                    # Download instead of buffering potentially large clips in memory.
                    try:
                        with clip.open("rb") as source:
                            self.send_response(200)
                            self.send_header("Content-Type", "video/mp4")
                            self.send_header("Content-Length", str(clip.stat().st_size))
                            self.send_header("Content-Disposition", f'attachment; filename="{clip.name}"')
                            self.send_header("X-Content-Type-Options", "nosniff")
                            self.end_headers()
                            while chunk := source.read(65536):
                                self.wfile.write(chunk)
                    except (OSError, ConnectionError):
                        pass
                else:
                    self._json({"error": "Clip not found."}, 404)
            elif path in ("/", "/app.js", "/style.css"):
                name, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                              "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                              "/style.css": ("style.css", "text/css; charset=utf-8")}[path]
                self._send((assets / name).read_bytes(), mime)
            else:
                self._json({"error": "Not found."}, 404)

        def do_POST(self):
            if not self._allowed() or not secrets.compare_digest(
                    self.headers.get("X-Trace-Token", ""), dashboard.token):
                self._json({"error": "Refresh the local dashboard and try again."}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request is missing or too large.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
                path = urlsplit(self.path).path
                if path == "/api/start":
                    result = dashboard.start(payload.get("mode", "monitor"))
                elif path == "/api/stop":
                    result = dashboard.stop()
                elif path == "/api/config":
                    result = dashboard.save(payload)
                else:
                    self._json({"error": "Not found."}, 404)
                    return
                self._json(result)
            except (ValueError, RuntimeError) as error:
                self._json({"error": str(error)}, 400)
            except OSError:
                self._json({"error": "Could not access the configuration file. Check permissions."}, 500)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
