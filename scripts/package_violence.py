"""Bundle the trained classifier and its matching encoder, without video data."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def package(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    files = ['runs/violence_baseline/best.pt', 'models/r3d_18-b3b3357e.pth']
    checksums = {}
    for name in files:
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f'Missing model component: {source}')
        checksums[name] = {'sha256': digest_file(source), 'bytes': source.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'format_version': 1, 'files': checksums,
                'instructions': 'Extract into the cloned repository root. See docs/FRIEND_SETUP.md. Python dependencies must be installed separately.'}
    # Exclusive mode prevents silently replacing a previously shared model release.
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for name in files:
            archive.write(root / name, name)
        archive.writestr('violence-model-manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(destination) as archive:
        if set(archive.namelist()) != set(files + ['violence-model-manifest.json']):
            raise ValueError('Unexpected archive content.')
        for name, entry in checksums.items():
            digest = hashlib.sha256()
            with archive.open(name) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Archive verification failed: {name}')
    print(json.dumps({'archive': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': digest_file(destination), 'contents': manifest}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/trace-violence-model.zip')
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], args.output)
