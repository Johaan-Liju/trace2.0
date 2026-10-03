"""Import the supplied RLVS archive, deduplicate bytes, and create provisional splits."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
from zipfile import ZipFile


def prepare(archive, output, seed=42, resume=False):
    import cv2
    output = Path(output).resolve()
    if output.exists() and (not resume or (output / 'manifest.csv').exists()):
        raise ValueError('Output already exists. Use a new directory; an existing split must not be silently replaced.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'videos').mkdir(exist_ok=True)
    records, rejected, duplicates, seen = [], [], [], {}
    conflicts = set()
    with ZipFile(archive) as z:
        entries = sorted((i for i in z.infolist() if not i.is_dir()), key=lambda i: i.filename)
        for index, item in enumerate(entries):
            path = PurePosixPath(item.filename)
            if path.suffix.lower() not in {'.mp4', '.avi'}:
                continue
            # Source names never become extraction paths. Only validated class + SHA filenames do.
            label = {'Violence': 1, 'NonViolence': 0}.get(path.parent.name)
            if label is None:
                raise ValueError(f'Unknown class folder: {item.filename}')
            if item.file_size > 512 * 1024 * 1024:
                raise ValueError(f'Unexpectedly large entry: {item.filename}')
            content = z.read(item)  # zipfile also verifies CRC
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen:
                if seen[digest]['label'] != label:
                    conflicts.add(digest)
                duplicates.append({'source': item.filename, 'same_as': seen[digest]['source'], 'sha256': digest})
                continue
            target = output / 'videos' / (digest + path.suffix.lower())
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Existing video checksum mismatch: {target}')
            else:
                target.write_bytes(content)
            cap = cv2.VideoCapture(str(target))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, frame = cap.read()
            finally:
                cap.release()
            row = {'path': target.relative_to(output).as_posix(), 'label': label,
                   'class': 'violence' if label else 'non_violence', 'sha256': digest,
                   'source': item.filename, 'group': digest}
            seen[digest] = row
            if not ok or not math.isfinite(fps) or fps <= 0 or count < 2:
                rejected.append({**row, 'reason': 'Cannot decode first frame or invalid FPS/frame count'})
                continue
            row.update(fps=round(fps, 6), frames=count, duration_seconds=round(count / fps, 3),
                       width=frame.shape[1], height=frame.shape[0])
            records.append(row)
            if len(records) % 100 == 0:
                print(f'Imported {len(records)} unique readable videos; archive entry {index + 1}/{len(entries)}', flush=True)
    rejected.extend({**r, 'reason': 'Identical bytes appear in both label classes'} for r in records if r['sha256'] in conflicts)
    records = [r for r in records if r['sha256'] not in conflicts]
    if not records:
        raise ValueError('No readable labeled videos found.')
    rng = random.Random(seed)
    for label in (0, 1):
        members = [r for r in records if r['label'] == label]
        if len(members) < 10:
            raise ValueError('Need at least 10 readable unique videos per class.')
        rng.shuffle(members)
        train_end, val_end = int(.7 * len(members)), int(.85 * len(members))
        for index, row in enumerate(members):
            row['split'] = 'train' if index < train_end else 'val' if index < val_end else 'test'
    with (output / 'manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: (r['split'], r['label'], r['sha256'])))
    report = {'archive': Path(archive).name, 'seed': seed, 'unique_readable_videos': len(records),
              'class_counts': dict(Counter(r['class'] for r in records)),
              'split_counts': dict(Counter(r['split'] + '/' + r['class'] for r in records)),
              'duplicates_removed': len(duplicates), 'duplicate_details': duplicates,
              'conflicting_label_videos_excluded': len(conflicts),
              'rejected': rejected, 'total_unique_hours': sum(r['duration_seconds'] for r in records) / 3600,
              'split_status': 'Provisional stratified video split. Exact duplicates removed. Original recording identities unavailable; near-duplicate or same-scene leakage has not been ruled out.',
              'label_scope': 'Whole-video violence/non-violence; no temporal intervals or stalking labels provided.'}
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'duplicate_details'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--output', default='data/violence')
    p.add_argument('--resume', action='store_true', help='Reuse checksum-verified files from an interrupted import')
    args = p.parse_args()
    prepare(args.archive, args.output, resume=args.resume)
