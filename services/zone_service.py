"""A small OpenCV editor: click polygon corners on a frozen camera frame."""

import json
import logging
import textwrap
from pathlib import Path

import cv2

from services.video_service import Camera
from services.utils import Utils


class ZoneEditor:
    """Edit one zone at a time, preserving the other camera settings."""

    WINDOW = "TRACE - Zone Editor"
    HELP = "Click 3+ corners in order, then click Save (or press S / Enter)."

    # ---------- Editor setup ----------

    def __init__(self, config_path, zone_id, name=None, zone_type=None):
        """Load an existing polygon or prepare a new one with this ID."""
        self.config_path = Path(config_path)
        self.config = Utils.load_camera_config(self.config_path)
        zones = Utils.get_zones(self.config)
        existing = next((zone for zone in zones if zone["id"] == zone_id), {})
        self.zone = dict(existing)
        self.zone.update({"id": zone_id,
                          "name": name if name is not None else existing.get("name", "Restricted Area"),
                          "type": zone_type if zone_type is not None else existing.get("type", "restricted")})
        self.points = [list(point) for point in existing.get("points", [])]
        self.other_zones = [zone for zone in zones if zone["id"] != zone_id]
        self.preview = None
        self.message = self.HELP
        self.pending_action = None

    def capture_preview(self):
        """Capture one frame, release the camera, and resize it for editing."""
        camera = Camera(self.config)
        try:
            if not camera.open():
                raise ConnectionError("Cannot open the camera for zone editing.")
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("No frame is available for zone editing.")
            self.preview = Utils.fit_frame(frame, min(self.config["display_width"], 960), 540)
        finally:
            camera.close()

    # ---------- Mouse input ----------

    def get_buttons(self):
        """Place clickable controls below the image, outside polygon coordinates."""
        height, width = self.preview.shape[:2]
        button_width = width // 4
        buttons = []
        for index, action in enumerate(("save", "undo", "clear", "cancel")):
            left = index * button_width + 4
            buttons.append((action, (left, height + 8, left + button_width - 8, height + 42)))
        return buttons

    def on_mouse(self, event, x, y, flags, parameter):
        """Convert a clicked preview pixel to a normalized polygon corner."""
        if self.preview is None:
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self.pending_action = "undo"
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for action, (left, top, right, bottom) in self.get_buttons():
            if left <= x <= right and top <= y <= bottom:
                self.pending_action = action
                return

        self.add_corner(x, y)

    def add_corner(self, x, y):
        """Add an image click as a corner, or close the outline near its start."""
        height, width = self.preview.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        if len(self.points) >= 3:
            first_x, first_y = self.points[0]
            distance_squared = (x - first_x * width) ** 2 + (y - first_y * height) ** 2
            if distance_squared <= 100:  # Within 10 pixels of the first corner.
                self.message = "Outline closed. Click Save or press S / Enter."
                return
        self.points.append([x / width, y / height])
        self.message = f"{len(self.points)} corners | {self.HELP}"
        logging.info("Added corner %s.", len(self.points))

    # ---------- Editor rendering ----------

    def draw_preview(self):
        """Combine saved zones, editable corners, and the control panel."""
        preview = Utils.draw_zones(self.preview, self.other_zones)
        preview = self.draw_corners(preview)
        return self.draw_controls(preview)

    def draw_corners(self, preview):
        """Draw the polygon being edited and number its corners in click order."""
        if not self.points:
            return preview
        draft = dict(self.zone, points=self.points)
        preview = Utils.draw_zones(preview, [draft])
        for index, (x, y) in enumerate(Utils.zone_contour(draft, preview.shape)):
            point = (round(float(x)), round(float(y)))
            cv2.circle(preview, point, 5, (255, 255, 255), -1)
            cv2.putText(preview, str(index + 1), (point[0] + 8, point[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return preview

    def draw_controls(self, preview):
        """Add the Save/Undo/Clear/Cancel buttons and the current help message."""
        # Put controls below the image so they do not hide polygon corners.
        height = preview.shape[0]
        preview = cv2.copyMakeBorder(preview, 0, 88, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        for action, (left, top, right, bottom) in self.get_buttons():
            color = (55, 110, 55) if action == "save" else (65, 65, 65)
            cv2.rectangle(preview, (left, top), (right, bottom), color, -1)
            cv2.putText(preview, action.capitalize(), (left + 7, top + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        lines = textwrap.wrap(self.message, width=max(20, preview.shape[1] // 7))
        for index, line in enumerate(lines[:2]):
            cv2.putText(preview, line, (8, height + 62 + index * 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return preview

    # ---------- Saving ----------

    def save_zone(self):
        """Validate and save just this zone, preserving the raw source settings."""
        zone = dict(self.zone, points=self.points)
        Utils.get_zones({"zones": [zone]})

        # Read the raw JSON again so a source override from the environment is
        # never accidentally written into the user's camera configuration.
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        zones = list(config.get("zones", []))
        for index, existing in enumerate(zones):
            if existing["id"] == zone["id"]:
                zones[index] = zone
                break
        else:
            zones.append(zone)
        config["zones"] = zones
        Utils.get_zones(config)
        Utils.save_json(self.config_path, config)

    # ---------- Window loop ----------

    def handle_action(self, action):
        """Handle a button or key; return True/False only when editing is done."""
        if action == "cancel":
            return False
        if action == "undo" and self.points:
            self.points.pop()
        if action == "clear":
            self.points.clear()
        if action in ("undo", "clear"):
            self.message = f"{len(self.points)} corners | {self.HELP}"
        if action == "save":
            try:
                self.save_zone()
                return True
            except ValueError as error:
                self.message = str(error)
                logging.warning("Zone not saved: %s", error)
        return None

    def run(self):
        """Open the editor and return True after saving, or False on cancel."""
        self.capture_preview()
        logging.info("Zone editor: click corners on the image, then click the green Save button.")
        key_actions = {10: "save", 13: "save", ord("s"): "save", ord("S"): "save",
                       8: "undo", 127: "undo", ord("r"): "clear",
                       ord("q"): "cancel", 27: "cancel"}
        try:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.WINDOW, self.on_mouse)
            while True:
                cv2.imshow(self.WINDOW, self.draw_preview())
                key = cv2.waitKey(20) & 0xFF
                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return False
                action = key_actions.get(key) or self.pending_action
                self.pending_action = None
                result = self.handle_action(action)
                if result is not None:
                    return result
        finally:
            cv2.destroyAllWindows()
