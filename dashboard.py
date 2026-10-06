"""Launch TRACE's local browser dashboard: python dashboard.py."""

import argparse
import logging
from pathlib import Path

from services.web_service import Dashboard, create_server


def main():
    parser = argparse.ArgumentParser(description="TRACE local browser dashboard")
    default = "config/camera.local.json"
    parser.add_argument("--config", default=default if Path(default).exists() else "config/camera.example.json")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dashboard = Dashboard(args.config)
    server = create_server(dashboard, args.port)
    logging.info("Open http://127.0.0.1:%s in your browser. Ctrl+C to exit.", args.port)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
        if dashboard.worker:
            dashboard.worker.join()
        server.server_close()


if __name__ == "__main__":
    main()
