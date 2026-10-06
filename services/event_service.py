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
        self.forget_expired_tracks(tracks, remembered_track_ids)
        entries = []

        for track in tracks:
            track_id = track["track_id"]
            current_zones = self.get_restricted_zones(track)
            previous_zones = self.inside_zones.get(track_id, set())

            # Inside now + not inside before = a new entry.
            for zone_id in self.restricted_zones:
                if zone_id in current_zones and zone_id not in previous_zones:
                    entries.append(self.create_entry(track, zone_id))
            self.inside_zones[track_id] = current_zones

        return entries

    def get_restricted_zones(self, track):
        """Return this person's restricted zone IDs; ignored objects return none."""
        if track["class"] != "person" or track.get("ignored", False):
            return set()
        zone_ids = set()
        for zone_id in track.get("zone_ids", []):
            if zone_id in self.restricted_zones:
                zone_ids.add(zone_id)
        return zone_ids

    def create_entry(self, track, zone_id):
        """Build one alert record for one person entering one zone."""
        return {
            "event_type": "RESTRICTED_ENTRY",
            "camera_id": self.camera_id,
            "track_id": track["track_id"],
            "zone_id": zone_id,
            "zone_name": self.restricted_zones[zone_id]["name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": track["confidence"],
        }

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

    # ---------- State cleanup ----------

    def forget_expired_tracks(self, tracks, remembered_track_ids):
        """Remove old IDs while preserving visible people and briefly lost people."""
        remembered = set(remembered_track_ids)
        for track in tracks:
            remembered.add(track["track_id"])
        for track_id in list(self.inside_zones):
            if track_id not in remembered:
                del self.inside_zones[track_id]

    def reset(self):
        """Clear stale memberships after a camera reconnect."""
        self.inside_zones.clear()
        self.last_message = ""
