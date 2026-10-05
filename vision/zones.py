"""Phase 4: validate camera zones and find which zones contain each person."""

import cv2

from vision.utils import TraceUtils


class ZoneManager:
    """Check polygon membership; event timers and alerts come in later phases."""

    ZONE_TYPES = ("restricted", "loitering", "crowd", "ignore")

    # ---------- Zone configuration ----------

    def __init__(self, config):
        """Load this camera's zones and reject invalid or duplicate polygons."""
        self.zones = config.get("zones", [])
        if not isinstance(self.zones, list):
            raise ValueError("'zones' must be a list.")

        used_ids = set()
        for zone in self.zones:
            if not isinstance(zone, dict):
                raise ValueError("Each zone must be a JSON object.")
            for field in ("id", "name", "type"):
                if not isinstance(zone.get(field), str) or not zone[field].strip():
                    raise ValueError(f"Zone '{field}' must be a non-empty string.")
            if zone["id"] in used_ids:
                raise ValueError(f"Duplicate zone ID: {zone['id']}")
            used_ids.add(zone["id"])
            if zone["type"] not in self.ZONE_TYPES:
                raise ValueError(f"Zone type must be one of: {', '.join(self.ZONE_TYPES)}.")
            TraceUtils.validate_polygon(zone.get("points"))

    # ---------- Polygon membership ----------

    def assign_zones(self, tracks, frame_shape):
        """Return copies of tracks with their current zone memberships."""
        polygons = [(zone, TraceUtils.zone_contour(zone, frame_shape)) for zone in self.zones]
        assigned_tracks = []

        for track in tracks:
            point = TraceUtils.get_foot_point(track["bbox"])
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

    def count_tracks(self, tracks):
        """Count current occupants; ignore zones override overlapping active zones."""
        counts = {zone["id"]: 0 for zone in self.zones}
        for track in tracks:
            for zone_id in track["zone_ids"]:
                counts[zone_id] += 1
        return counts
