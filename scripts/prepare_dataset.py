"""Build a person-only YOLO dataset from a reviewed, grouped CSV manifest."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


def prepare(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists. Choose a new directory to avoid overwriting data.')
    records, groups, hashes = [], {}, {}
    counts = dict.fromkeys(('train', 'val', 'test'), 0)
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'image', 'label', 'group', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('CSV needs image,label,group,split columns.')
        for line, row in enumerate(reader, 2):
            split, group = row['split'].strip(), row['group'].strip()
            if split not in counts or not group:
                raise ValueError(f'Row {line}: use train/val/test and a nonempty recording group.')
            if group in groups and groups[group] != split:
                raise ValueError(f'Group {group} crosses splits. Keep a camera session together.')
            groups[group] = split
            image = (manifest.parent / row['image']).resolve()
            label = (manifest.parent / row['label']).resolve()
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png'} or not image.is_file() or not label.is_file():
                raise ValueError(f'Row {line}: image or label is missing/unsupported.')
            # Empty labels are explicit, reviewed negative frames. Missing labels are errors.
            for box in label.read_text(encoding='utf-8').splitlines():
                fields = box.split()
                if len(fields) != 5 or fields[0] != '0':
                    raise ValueError(f'{label}: expected class 0 and four coordinates.')
                x, y, w, h = map(float, fields[1:])
                if not all(math.isfinite(v) for v in (x, y, w, h)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'{label}: invalid normalized bounding box.')
                if min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1.00001:
                    raise ValueError(f'{label}: box extends outside the image.')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f'Duplicate image at row {line}, previously row {hashes[digest]}.')
            hashes[digest] = line
            records.append((image, label, split, group))
            counts[split] += 1
    if any(n == 0 for n in counts.values()):
        raise ValueError('Provide nonempty train, val, and test sets from separate recording groups.')
    for split in counts:
        for kind in ('images', 'labels'):
            (output / kind / split).mkdir(parents=True)
    for index, (image, label, split, group) in enumerate(records):
        stem = f'{index:06d}'
        shutil.copy2(image, output / 'images' / split / (stem + image.suffix.lower()))
        shutil.copy2(label, output / 'labels' / split / (stem + '.txt'))
    yaml = f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n'
    (output / 'dataset.yaml').write_text(yaml, encoding='utf-8')
    shutil.copy2(manifest, output / 'source_manifest.csv')
    print(json.dumps({'counts': counts, 'recording_groups': len(groups), 'yaml': str(output / 'dataset.yaml')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='data/person_dataset')
    args = parser.parse_args()
    try:
        prepare(args.manifest, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Dataset: {exc}\n')

