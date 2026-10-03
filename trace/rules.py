"""Event logic independent of the detector, UI, and wall-clock speed."""
from dataclasses import dataclass, field
import math


def validate_polygon(points):
    if len(points) < 3:
        raise ValueError("A zone needs at least three points.")
    polygon = [(float(x), float(y)) for x, y in points]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for p in polygon for v in p):
        raise ValueError("Zone coordinates must be finite numbers between 0 and 1.")
    area = sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(polygon, polygon[1:] + polygon[:1]))
    if abs(area) < 1e-6:
        raise ValueError("Zone has no usable area. Select points around its boundary.")
    return polygon


def inside(point, polygon):
    """Ray casting, including boundary points."""
    x, y = point
    result = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-9 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            result = not result
    return result


@dataclass
class Presence:
    observed_seconds: float = 0.0
    last_seen: float = 0.0
    alerted: bool = False


@dataclass
class ZoneRule:
    polygon: list
    dwell_seconds: float = 1.0
    missing_timeout: float = 1.0
    states: dict = field(default_factory=dict)
    last_time: float = -1.0

    def __post_init__(self):
        self.polygon = validate_polygon(self.polygon)
        if not math.isfinite(self.dwell_seconds) or self.dwell_seconds < 0:
            raise ValueError("Dwell seconds must be finite and nonnegative.")
        if not math.isfinite(self.missing_timeout) or self.missing_timeout <= 0:
            raise ValueError("Missing timeout must be finite and positive.")

    def update(self, timestamp, tracks):
        """tracks: {temporary_id: normalized bottom-center point}.

        Only consecutive observations contribute dwell time. Brief missing
        detections preserve state without counting the unseen interval.
        Leaving the zone resets the visit. One alert per visit / track ID.
        """
        if not math.isfinite(timestamp) or timestamp < self.last_time:
            raise ValueError("Timestamps must be finite and nondecreasing.")
        previous_time = self.last_time
        self.last_time = timestamp
        for track_id in list(self.states):
            if timestamp - self.states[track_id].last_seen > self.missing_timeout:
                del self.states[track_id]
        alerts = []
        for track_id, point in tracks.items():
            if not inside(point, self.polygon):
                self.states.pop(track_id, None)
                continue
            state = self.states.setdefault(track_id, Presence(last_seen=timestamp))
            if state.last_seen == previous_time:
                state.observed_seconds += timestamp - previous_time
            state.last_seen = timestamp
            if not state.alerted and state.observed_seconds >= self.dwell_seconds:
                state.alerted = True
                alerts.append({"kind": "restricted_area_entry", "track_id": track_id,
                               "source_seconds": round(timestamp, 3),
                               "observed_seconds": round(state.observed_seconds, 3)})
        return alerts

