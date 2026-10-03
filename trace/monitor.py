"""Run person detection + ByteTrack + a user-selected restricted zone."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

from trace.rules import ZoneRule, validate_polygon


def choose_zone(frame, cv2, np):
    height, width = frame.shape[:2]
    scale = min(1.0, 1100 / width, 720 / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
    ph, pw = preview.shape[:2]
    points = []
    title = "TRACE - click zone corners, Enter: save, R: reset, Esc: cancel"
    cv2.namedWindow(title)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((min(1., max(0., x / pw)), min(1., max(0., y / ph))))

    cv2.setMouseCallback(title, click)
    try:
        while True:
            canvas = preview.copy()
            pixels = np.array([(int(x * pw), int(y * ph)) for x, y in points], np.int32)
            for p in pixels:
                cv2.circle(canvas, tuple(p), 5, (0, 210, 255), -1)
            if len(pixels) > 1:
                cv2.polylines(canvas, [pixels], len(pixels) >= 3, (0, 210, 255), 2)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Zone selection cancelled.")
            if key in (ord('r'), ord('R')):
                points.clear()
            if key in (10, 13):
                try:
                    return validate_polygon(points)
                except ValueError as exc:
                    print(exc)
    finally:
        cv2.destroyWindow(title)


def run(args):
    config_dir = Path('.runtime/ultralytics').resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    import cv2
    import numpy as np
    from ultralytics import YOLO

    is_file = Path(args.source).is_file()
    source = int(args.source) if args.source.isdecimal() and not is_file else args.source
    model = YOLO(args.model)
    person_ids = [int(k) for k, value in model.names.items() if value == "person"]
    if not person_ids:
        raise ValueError("The model needs a class named 'person'.")
    capture = cv2.VideoCapture(source)
    database = None
    writer = None
    frames = 0
    events_count = 0
    started = time.perf_counter()
    run_dir = Path(args.output) / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Cannot read the source. Check the file, camera index, or stream connection.")
        height, width = frame.shape[:2]
        if args.select_zone:
            polygon = choose_zone(frame, cv2, np)
            Path(args.zone).write_text(json.dumps({"polygon": polygon}, indent=2), encoding='utf-8')
        else:
            polygon = json.loads(Path(args.zone).read_text(encoding='utf-8'))["polygon"]
        rule = ZoneRule(polygon, args.dwell, args.missing_timeout)
        run_dir.mkdir(parents=True)
        (run_dir / 'snapshots').mkdir()
        # Do not persist camera URLs, which can contain credentials.
        (run_dir / 'settings.json').write_text(json.dumps({
            'source_type': 'file' if is_file else 'live', 'polygon': rule.polygon,
            'dwell_seconds': args.dwell, 'confidence': args.conf, 'model': Path(args.model).name
        }, indent=2), encoding='utf-8')
        database = sqlite3.connect(run_dir / 'events.sqlite3')
        database.execute('CREATE TABLE events (id TEXT PRIMARY KEY, created_utc TEXT, kind TEXT, track_id INTEGER, source_seconds REAL, observed_seconds REAL, snapshot TEXT)')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            if is_file:
                raise ValueError("Video FPS is unavailable; convert the video to constant frame rate first.")
            fps = 25.0
        if args.save_video:
            writer = cv2.VideoWriter(str(run_dir / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                raise ValueError("Cannot create the annotated video.")
        pixels = np.array([(round(x * (width - 1)), round(y * (height - 1))) for x, y in rule.polygon], np.int32)
        started = time.perf_counter()
        last_timestamp = -1.0
        with (run_dir / 'events.jsonl').open('w', encoding='utf-8') as event_file:
            while ok:
                # File time must not depend on inference speed; use media PTS.
                if is_file:
                    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    if not math.isfinite(timestamp) or timestamp <= last_timestamp:
                        timestamp = max(frames / fps, last_timestamp + 1 / fps)
                else:
                    timestamp = time.perf_counter() - started
                last_timestamp = timestamp
                result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                     classes=person_ids, conf=args.conf, imgsz=args.imgsz,
                                     device=args.device, verbose=False)[0]
                tracks = {}
                if result.boxes is not None and result.boxes.id is not None:
                    for box, identity in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist()):
                        x1, y1, x2, y2 = box
                        tracks[int(identity)] = ((x1 + x2) / (2 * width), y2 / height)
                alerts = rule.update(timestamp, tracks)
                annotated = result.plot()
                cv2.polylines(annotated, [pixels], True, (0, 210, 255), 3)
                active = sum(s.last_seen == timestamp for s in rule.states.values())
                label = f"RESTRICTED ZONE: {active} person(s) | alerts: {events_count + len(alerts)} | Q: quit"
                cv2.putText(annotated, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 210, 255), 2)
                for event in alerts:
                    event['id'] = uuid.uuid4().hex
                    event['created_utc'] = datetime.now(timezone.utc).isoformat()
                    event['snapshot'] = f"snapshots/{event['id']}.jpg"
                    if not cv2.imwrite(str(run_dir / event['snapshot']), annotated):
                        raise OSError("Could not save event snapshot.")
                    database.execute('INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)',
                                     tuple(event[k] for k in ['id', 'created_utc', 'kind', 'track_id', 'source_seconds', 'observed_seconds', 'snapshot']))
                    database.commit()
                    event_file.write(json.dumps(event) + '\n')
                    event_file.flush()
                    print(json.dumps(event), flush=True)
                    events_count += 1
                if writer:
                    writer.write(annotated)
                frames += 1
                if not args.headless:
                    scale = min(1.0, 1200 / width, 800 / height)
                    cv2.imshow('TRACE - monitoring', cv2.resize(annotated, (round(width * scale), round(height * scale))))
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    if cv2.getWindowProperty('TRACE - monitoring', cv2.WND_PROP_VISIBLE) < 1:
                        break
                if args.max_frames and frames >= args.max_frames:
                    break
                ok, frame = capture.read()
            if not ok and not is_file:
                raise RuntimeError("Live input stopped. Monitoring has ended; restart after restoring the connection.")
    finally:
        capture.release()
        if writer:
            writer.release()
        if database:
            database.close()
        cv2.destroyAllWindows()
    seconds = time.perf_counter() - started
    summary = {'frames': frames, 'events': events_count, 'processing_seconds': round(seconds, 2),
               'processing_fps': round(frames / max(seconds, .001), 2), 'output': str(run_dir)}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Video path, webcam index (0), or RTSP URL')
    parser.add_argument('--model', default='models/yolo11n.pt')
    parser.add_argument('--zone', default='zone.local.json')
    parser.add_argument('--select-zone', action='store_true')
    parser.add_argument('--dwell', type=float, default=1.0)
    parser.add_argument('--missing-timeout', type=float, default=1.0)
    parser.add_argument('--conf', type=float, default=.35)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu', help='cpu or 0 for a CUDA GPU')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()
    if args.headless and args.select_zone:
        parser.error('--select-zone needs a visible window')
    if not 0 < args.conf <= 1 or args.imgsz <= 0 or args.max_frames < 0:
        parser.error('Confidence must be in (0,1], imgsz positive, max-frames nonnegative')
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'TRACE: {exc}\n')


if __name__ == '__main__':
    main()

