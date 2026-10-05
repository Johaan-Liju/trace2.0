"""Immediate alerts when a tracked person enters a restricted zone."""

import logging
from datetime import datetime, timezone

from services.utils import Utils


class EntryAlerts:
    """Remember current memberships so each entry produces one alert."""

    # ---------- Alert setup ----------

    def __init__(self, config, zones):
        """Select restricted zones and prepare the local alert output."""
        self.camera_id = config["camera_id"]
        self.restricted_zones = {zone["id"]: zone for zone in zones if zone["type"] == "restricted"}
        if not self.restricted_zones:
            raise ValueError("Add a restricted zone with edit_zones.py before enabling alerts.")
        settings = config.get("alerts", {})
        if not isinstance(settings, dict) or type(settings.get("sound", True)) is not bool:
            raise ValueError("'alerts' must be an object with a true/false 'sound' setting.")
        self.sound = settings.get("sound", True)
        self.inside_zones = {}
        self.last_message = ""

    # ---------- Entry detection ----------

    def check_entries(self, tracks, remembered_track_ids):
        """Return new entries immediately; do not repeat for continuous presence."""
        remembered = set(remembered_track_ids) | {track["track_id"] for track in tracks}
        self.inside_zones = {track_id: zones for track_id, zones in self.inside_zones.items()
                             if track_id in remembered}
        entries = []

        for track in tracks:
            track_id = track["track_id"]
            current = set()
            if track["class"] == "person" and not track.get("ignored", False):
                current = set(track.get("zone_ids", [])) & self.restricted_zones.keys()
            previous = self.inside_zones.get(track_id, set())

            for zone_id in self.restricted_zones:
                if zone_id in current and zone_id not in previous:
                    entries.append({
                        "event_type": "RESTRICTED_ENTRY",
                        "camera_id": self.camera_id,
                        "track_id": track_id,
                        "zone_id": zone_id,
                        "zone_name": self.restricted_zones[zone_id]["name"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "confidence": track["confidence"],
                    })
            self.inside_zones[track_id] = current

        return entries

    # ---------- Local notification ----------

    def notify(self, entries):
        """Log each entry, remember the latest message, and play one sound."""
        for entry in entries:
            time_label = entry["timestamp"][11:19] + " UTC"
            self.last_message = (f"Last entry {time_label}: person #{entry['track_id']} "
                                 f"in {entry['zone_name']}")
            logging.warning("RESTRICTED ENTRY | %s | person #%s | %s | %s",
                            entry["camera_id"], entry["track_id"], entry["zone_name"], entry["timestamp"])
        if entries and self.sound:
            Utils.play_alert_sound()

    def reset(self):
        """Clear stale memberships after a camera reconnect."""
        self.inside_zones.clear()
        self.last_message = ""
