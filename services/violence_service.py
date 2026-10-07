"""Queue video scans in an isolated process so HTTP and camera capture keep working."""

import copy
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone


class ViolenceScans:
    MAX_PENDING = 16
    MAX_HISTORY = 100
    MAX_UPLOAD = 256 * 1024 * 1024
    EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    def __init__(self, root=None, storage=None):
        self.root = Path(root or Path(__file__).resolve().parent.parent).resolve()
        self.storage = Path(storage or self.root / 'storage/violence').resolve()
        self.lock = threading.RLock()
        self.jobs = {}
        self.thread = None
        self.closed = False

    def availability(self):
        missing = [name for name in ('runs/violence_baseline/best.pt', 'models/r3d_18-b3b3357e.pth')
                   if not (self.root / name).is_file()]
        return {'ready': not missing, 'missing': missing, 'max_upload_bytes': self.MAX_UPLOAD}

    def enqueue(self, path, name=None, clip_id=None):
        with self.lock:
            if self.closed:
                raise ValueError('The dashboard is shutting down.')
            if not self.availability()['ready']:
                raise ValueError('Violence model files are missing. Add the classifier and matching encoder first.')
            path = Path(path).resolve()
            if not path.is_file() or path.suffix.lower() not in self.EXTENSIONS:
                raise ValueError('Choose an existing MP4, AVI, MOV, MKV, or WebM video.')
            if clip_id:
                existing = next((job for job in reversed(list(self.jobs.values()))
                                 if job['clip_id'] == clip_id and job['status'] not in ('error', 'cancelled')), None)
                if existing:
                    return existing['id']
            if sum(job['status'] in ('queued', 'running', 'cancelling') for job in self.jobs.values()) >= self.MAX_PENDING:
                raise ValueError('Scan queue is full. Wait for a scan to finish or cancel one.')
            while len(self.jobs) >= self.MAX_HISTORY:
                oldest = next((key for key, job in self.jobs.items()
                               if job['status'] in ('complete', 'error', 'cancelled')), None)
                if oldest is None:
                    raise ValueError('Scan history is full.')
                del self.jobs[oldest]
            key = secrets.token_hex(12)
            self.jobs[key] = {'id': key, 'name': name or path.name, 'clip_id': clip_id,
                              'created_at': datetime.now(timezone.utc).isoformat(),
                              'status': 'queued', 'done': 0, 'total': 0, 'error': '', 'summary': None,
                              '_source': path, '_cancel': threading.Event()}
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            return key

    def snapshot(self):
        with self.lock:
            return {**self.availability(), 'jobs': [copy.deepcopy({key: value for key, value in job.items()
                    if not key.startswith('_')}) for job in reversed(list(self.jobs.values()))]}

    def cancel(self, key):
        with self.lock:
            if not isinstance(key, str):
                raise ValueError('Choose a scan to cancel.')
            job = self.jobs.get(key)
            if job is None:
                raise ValueError('Scan not found.')
            if job['status'] in ('queued', 'running', 'cancelling'):
                job['_cancel'].set()
                job['status'] = 'cancelled' if job['status'] == 'queued' else 'cancelling'

    def file(self, key, kind):
        with self.lock:
            job = self.jobs.get(key)
            if not job:
                return None
            if kind == 'video':
                return job['_source']
            if kind == 'report' and job['status'] == 'complete':
                return self.storage / f'{key}.json'
        return None

    def close(self):
        with self.lock:
            self.closed = True
            for key in self.jobs:
                self.cancel(key)
            worker = self.thread
        if worker:
            worker.join(timeout=10)

    def _run(self):
        while True:
            with self.lock:
                job = next((job for job in self.jobs.values() if job['status'] == 'queued'), None)
                if job is None or self.closed:
                    self.thread = None
                    return
                job['status'] = 'running'
            self._execute(job)

    def _execute(self, job):
        process = None
        report = self.storage / f"{job['id']}.json"
        try:
            self.storage.mkdir(parents=True, exist_ok=True)
            log = self.storage / f"{job['id']}.log"
            command = [sys.executable, '-u', '-m', 'trace.predict_violence', '--source', str(job['_source']),
                       '--scan', '--output', str(report)]
            with log.open('w', encoding='utf-8') as output:
                process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.DEVNULL, stderr=output,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                with log.open(encoding='utf-8', errors='replace') as progress:
                    while True:
                        if job['_cancel'].is_set():
                            if process.poll() is None:
                                process.terminate()
                            process.wait(timeout=5)
                            with self.lock:
                                job['status'] = 'cancelled'
                            return
                        for line in progress.readlines():
                            match = re.search(r'Scanned (\d+)/(\d+) windows', line)
                            if match:
                                with self.lock:
                                    job['done'], job['total'] = map(int, match.groups())
                        code = process.poll()
                        if code is not None:
                            break
                        job['_cancel'].wait(0.25)
            if code != 0:
                raise RuntimeError('Scan process failed')
            result = json.loads(report.read_text(encoding='utf-8'))
            with self.lock:
                if job['_cancel'].is_set():
                    job['status'] = 'cancelled'
                else:
                    job['done'] = job['total'] = result['windows_scanned']
                    job['summary'] = {key: result[key] for key in ('prediction', 'violence_score', 'threshold',
                        'duration_seconds', 'windows_scanned', 'review_segments', 'threshold_validated_for_scan')}
                    job['status'] = 'complete'
        except Exception:
            with self.lock:
                job['status'] = 'cancelled' if job['_cancel'].is_set() else 'error'
                if job['status'] == 'error':
                    job['error'] = 'Scan failed. Check that the video is readable, model files match, and PyTorch is installed. Details are in storage/violence.'
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if job['status'] != 'complete':
                report.unlink(missing_ok=True)
