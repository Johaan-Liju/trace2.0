"""Sample frames for human annotation. Keep one recording session per output folder."""
import argparse
import math
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--every-seconds', type=float, default=2.)
    p.add_argument('--limit', type=int, default=300)
    args = p.parse_args()
    if not math.isfinite(args.every_seconds) or args.every_seconds <= 0 or args.limit <= 0:
        p.error('Sampling interval and limit must be positive.')
    output = Path(args.output)
    if output.exists():
        p.error('Choose a new output folder to avoid overwriting frames.')
    cap = cv2.VideoCapture(args.video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            p.error('Cannot read this video or its FPS.')
        output.mkdir(parents=True)
        step, index, count = max(1, round(fps * args.every_seconds)), 0, 0
        while count < args.limit:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                if not cv2.imwrite(str(output / f'frame_{index:08d}.jpg'), frame):
                    raise OSError('Cannot save frame.')
                count += 1
            index += 1
        print(f'Saved {count} frames to {output}. Annotate all people before training.')
    finally:
        cap.release()


if __name__ == '__main__':
    main()

