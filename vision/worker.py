"""Camera -> optional YOLO -> optional ByteTrack -> zones -> labeled preview."""

import logging
import time

import cv2

from vision.camera import Camera
from vision.detector import Detector
from vision.tracker import Tracker
from vision.utils import TraceUtils
from vision.zones import ZoneManager


# ---------- Video pipeline ----------

def run_video(config, headless=False, max_frames=None, snapshot_path=None, detect=False, track=False):
    """Read video until EOF, Q, Ctrl+C, or the requested frame limit."""
    camera = Camera(config)
    frame_number = 0

    try:
        zone_manager = ZoneManager(config)
        detector = Detector(config) if detect or track else None
        if not camera.open():
            raise ConnectionError("Cannot open camera. Check the source and connection.")

        frame_rate = camera.get_fps()
        frame_interval = 1 / frame_rate
        tracker = Tracker(config, frame_rate=frame_rate) if track else None
        if tracker is not None:
            logging.info("Tracking remembers missing people for up to %s processed frames.",
                         tracker.settings["track_buffer"])
        connection_id = camera.connection_id
        logging.info("Camera online. Press Q in the preview or Ctrl+C to stop.")

        while True:
            frame_started = time.monotonic()
            frame = camera.read_frame()
            if frame is None:
                if frame_number == 0:
                    raise RuntimeError("Video opened but no frames could be decoded.")
                logging.info("Video ended or decoder stopped after %s frames.", frame_number)
                break

            frame_number += 1

            if tracker is not None and camera.connection_id != connection_id:
                tracker.reset()
                connection_id = camera.connection_id
                logging.info("Camera reconnected. Starting fresh tracking IDs.")

            # Detect on the original frame. Draw before resizing so boxes align.
            objects = []
            if tracker is not None:
                detections = detector.detect(frame, confidence=tracker.settings["track_low_thresh"])
                timestamp = time.monotonic()
                if camera.source_type == "file":
                    timestamp = (frame_number - 1) * frame_interval
                objects = tracker.update(detections, frame.shape, timestamp)
            elif detector is not None:
                objects = detector.detect(frame)

            # Membership uses original coordinates. Resizing happens afterward.
            annotated = frame
            if zone_manager.zones:
                objects = zone_manager.assign_zones(objects, frame.shape)
                counts = zone_manager.count_tracks(objects) if detector is not None else None
                annotated = TraceUtils.draw_zones(frame, zone_manager.zones, counts)
            if detector is not None:
                annotated = TraceUtils.draw_detections(annotated, objects)

            preview = TraceUtils.resize_frame(annotated, config["display_width"])
            preview = TraceUtils.add_preview_label(
                preview, config["camera_id"], frame_number
            )

            if snapshot_path and frame_number == 1:
                TraceUtils.save_snapshot(preview, snapshot_path)

            if not headless:
                cv2.imshow("TRACE - Camera Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("TRACE - Camera Preview", cv2.WND_PROP_VISIBLE) < 1:
                    break

            if max_frames is not None and frame_number >= max_frames:
                break

            # Local video files decode quickly; pace them at their recorded FPS.
            if camera.source_type == "file":
                remaining = frame_interval - (time.monotonic() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)

        return frame_number

    finally:
        camera.close()
        if not headless:
            cv2.destroyAllWindows()
        logging.info("Camera released.")
