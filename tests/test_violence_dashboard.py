"""Exercise scan queuing, uploads, media seeking, and camera integration."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from services.violence_service import ViolenceScans
from services.web_service import Dashboard, create_server


class ViolenceDashboardTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for name in ('runs/violence_baseline/best.pt', 'models/r3d_18-b3b3357e.pth'):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'test')
        config = self.root / 'camera.json'
        config.write_text(json.dumps({'camera_id': 'test', 'name': 'Test', 'source': '0', 'zones': []}))
        self.dashboard = Dashboard(config)
        self.scans = ViolenceScans(self.root, self.root / 'scans')
        self.dashboard.scans = self.scans
        self.addCleanup(self.scans.close)
        self.video = self.root / 'test.mp4'
        self.video.write_bytes(b'0123456789')

    def http(self):
        server = create_server(self.dashboard, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f'http://127.0.0.1:{server.server_port}'

    def request(self, path, payload=None, raw=None, headers=None):
        data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        return urlopen(Request(self.base + path, data=data, headers={
            'X-Trace-Token': self.dashboard.token, **(headers or {})}), timeout=5)

    def test_upload_is_registered_without_exposing_paths_and_supports_seeking(self):
        self.http()
        with patch.object(self.scans, '_run'):
            with self.request('/api/violence/upload', raw=b'0123456789',
                              headers={'X-File-Name': '..%2F..%2Fsample.mp4'}) as response:
                state = json.load(response)
        job = state['violence']['jobs'][0]
        self.assertEqual(job['name'], 'sample.mp4')
        self.assertNotIn('_source', job)
        self.assertNotIn(str(self.root), json.dumps(state))
        saved = self.scans.file(job['id'], 'video')
        self.assertEqual(saved.parent, self.scans.storage / 'uploads')
        self.assertEqual(saved.read_bytes(), b'0123456789')
        for range_header, expected in [('bytes=2-5', b'2345'), ('bytes=-3', b'789'), ('bytes=7-', b'789')]:
            with self.request(f"/api/violence/{job['id']}/video", headers={'Range': range_header}) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)
        with self.assertRaises(HTTPError) as caught:
            self.request(f"/api/violence/{job['id']}/video", headers={'Range': 'bytes=50-100'})
        self.assertEqual(caught.exception.code, 416)
        caught.exception.close()
        for route in ('/api/violence/unknown/video', '/api/violence/../../camera.json',
                      f"/api/violence/{job['id']}/report"):
            with self.assertRaises(HTTPError) as caught:
                self.request(route)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()

    def test_rejected_uploads_leave_no_file_and_require_token(self):
        self.http()
        for headers in ({'X-Trace-Token': ''}, {'Origin': 'https://example.com'}, {'X-File-Name': 'script.html'}):
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/violence/upload', raw=b'bad', headers=headers)
            self.assertIn(caught.exception.code, (400, 403))
            caught.exception.close()
        (self.root / 'models/r3d_18-b3b3357e.pth').unlink()
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/violence/upload', raw=b'bad', headers={'X-File-Name': 'video.mp4'})
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()
        self.assertEqual(list((self.scans.storage / 'uploads').glob('*')), [])

    def test_camera_clip_automatic_scan_and_manual_deduplication(self):
        with self.assertRaisesRegex(ValueError, 'Record entry clips'):
            self.dashboard.save({'auto_scan': True})
        self.dashboard.save({'clips': True, 'auto_scan': True})
        with patch.object(self.scans, '_run'):
            self.dashboard._add_clip(self.video)
            key = next(iter(self.dashboard.clip_paths))
            self.dashboard.scan_clip(key)
        self.assertEqual(len(self.scans.jobs), 1)
        self.assertEqual(next(iter(self.scans.jobs.values()))['clip_id'], key)
        self.assertTrue(self.dashboard.snapshot()['config']['auto_scan'])

    def test_auto_scan_failure_does_not_break_recording(self):
        self.dashboard.save({'clips': True, 'auto_scan': True})
        (self.root / 'models/r3d_18-b3b3357e.pth').unlink()
        self.dashboard._add_clip(self.video)
        state = self.dashboard.snapshot()
        self.assertEqual(len(state['clips']), 1)
        self.assertIn('missing', state['clips'][0]['scan_error'])

    def test_queue_is_bounded_and_queued_cancellation_allows_retry(self):
        with patch.object(self.scans, '_run'):
            for _ in range(self.scans.MAX_PENDING):
                self.scans.enqueue(self.video)
            with self.assertRaisesRegex(ValueError, 'queue is full'):
                self.scans.enqueue(self.video)
            key = next(iter(self.scans.jobs))
            self.scans.cancel(key)
            self.assertEqual(self.scans.jobs[key]['status'], 'cancelled')
            self.scans.enqueue(self.video)
            self.assertEqual(len(self.scans.jobs), self.scans.MAX_PENDING + 1)

    def test_running_scan_can_be_cancelled_and_process_is_reaped(self):
        entered = threading.Event()
        with patch('services.violence_service.subprocess.Popen') as popen:
            process = popen.return_value
            process.poll.return_value = None
            popen.side_effect = lambda *args, **kwargs: (entered.set(), process)[1]
            key = self.scans.enqueue(self.video)
            self.assertTrue(entered.wait(3))
            self.scans.cancel(key)
            self.scans.close()
            self.assertEqual(self.scans.jobs[key]['status'], 'cancelled')
            process.terminate.assert_called_once()
            self.assertTrue(process.wait.called)

    def test_worker_publishes_report_and_advances_after_failure(self):
        report_data = {'prediction': 'possible_violence', 'violence_score': .7, 'threshold': .3,
                       'duration_seconds': 2, 'windows_scanned': 3, 'threshold_validated_for_scan': False,
                       'review_segments': [{'start_seconds': 0, 'end_seconds': 1, 'peak_score': .7}]}
        with patch('services.violence_service.subprocess.Popen') as popen:
            process = popen.return_value
            process.poll.return_value = 0

            def finish(command, **kwargs):
                Path(command[-1]).write_text(json.dumps(report_data))
                kwargs['stderr'].write('Scanned 3/3 windows\n')
                kwargs['stderr'].flush()
                return process

            popen.side_effect = finish
            key = self.scans.enqueue(self.video)
            deadline = time.monotonic() + 4
            while self.scans.snapshot()['jobs'][0]['status'] not in ('complete', 'error') and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(self.scans.jobs[key]['status'], 'complete')
            self.assertEqual(self.scans.jobs[key]['done'], 3)
            self.assertTrue(self.scans.file(key, 'report').is_file())
            process.poll.return_value = 1
            failed = self.scans.enqueue(self.video)
            deadline = time.monotonic() + 4
            while self.scans.jobs[failed]['status'] not in ('complete', 'error') and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(self.scans.jobs[failed]['status'], 'error')
            self.assertIsNone(self.scans.file(failed, 'report'))
            process.poll.return_value = 0
            retried = self.scans.enqueue(self.video)
            deadline = time.monotonic() + 4
            while self.scans.jobs[retried]['status'] not in ('complete', 'error') and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(self.scans.jobs[retried]['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
