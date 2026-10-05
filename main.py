"""Run TRACE restricted-area monitoring: python main.py --help."""

import argparse
import logging

import cv2

from services.utils import Utils
from services.video_service import run_video


# ---------- Command-line settings ----------

def read_arguments():
    """Read the config path and optional preview / smoke-test settings."""
    parser = argparse.ArgumentParser(description="TRACE: restricted-area entry alerts")
    parser.add_argument("--config", default="config/camera.example.json")
    parser.add_argument("--headless", action="store_true", help="Run without a preview window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--snapshot", help="Save the first labeled frame to an image path")
    parser.add_argument("--detect", action="store_true", help="Enable YOLO person detection")
    parser.add_argument("--track", action="store_true", help="Enable YOLO and ByteTrack IDs")
    parser.add_argument("--alerts", action="store_true", help="Monitor restricted zones and alert immediately on entry")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    return args


# ---------- Application entry point ----------

def main():
    """Load camera settings, start the video loop, and report failures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = read_arguments()

    try:
        config = Utils.load_camera_config(args.config)
        frame_count = run_video(config, args.headless, args.max_frames, args.snapshot,
                                detect=args.detect, track=args.track, alerts=args.alerts)
        logging.info("Finished. Frames read: %s", frame_count)
        return 0
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
        return 0
    except cv2.error:
        # OpenCV exceptions can contain source URLs; do not print credentials.
        logging.error("OpenCV failed. Check the codec, camera, or preview window support.")
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
