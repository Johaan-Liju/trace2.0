"""Draw a camera zone: python edit_zones.py --config config/camera.local.json."""

import argparse
import logging

import cv2

from vision.zone_editor import ZoneEditor
from vision.zones import ZoneManager


# ---------- Command-line entry point ----------

def main():
    """Read zone details and open the mouse editor without loading YOLO."""
    parser = argparse.ArgumentParser(description="TRACE polygon zone editor")
    parser.add_argument("--config", default="config/camera.local.json")
    parser.add_argument("--zone-id", default="restricted_01", help="Reuse an ID to edit its polygon")
    parser.add_argument("--name", help="Zone label; existing names are kept when omitted")
    parser.add_argument("--type", choices=ZoneManager.ZONE_TYPES, dest="zone_type")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        editor = ZoneEditor(args.config, args.zone_id, args.name, args.zone_type)
        saved = editor.run()
        logging.info("Zone saved. Restart monitoring to use it." if saved else "Editing canceled.")
        return 0
    except KeyboardInterrupt:
        logging.info("Editing canceled.")
        return 0
    except cv2.error:
        logging.error("OpenCV could not run the editor. Close other camera previews and check desktop window support.")
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
