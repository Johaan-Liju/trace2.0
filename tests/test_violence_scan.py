"""Check temporal coverage and scoring without claiming model accuracy."""

from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from trace.predict_violence import merge_review_segments, predict, scan
from trace.violence import RECIPE, iter_video_windows, sample_video


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def video(self, count=120, fps=15):
        path = self.root / f'video-{count}-{fps}.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (32, 32))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(count):
                # A brief visual event between the baseline's three windows.
                writer.write(np.full((32, 32, 3), 255 if 30 <= index < 43 else 0, dtype=np.uint8))
        finally:
            writer.release()
        return path

    def checkpoint(self):
        head = torch.nn.Linear(512, 1)
        with torch.no_grad():
            head.weight.zero_()
            head.weight[0, 0] = 8
            head.bias.zero_()
        path = self.root / 'test.pt'
        torch.save({'recipe': RECIPE, 'encoder_sha256': 'test-encoder', 'head': head.state_dict(),
                    'clips': 3, 'mean': torch.zeros(512), 'std': torch.ones(512), 'threshold': .3}, path)
        return path

    @staticmethod
    def fake_encoder(batch):
        features = torch.zeros((batch.shape[0], 512))
        features[:, 0] = batch[:, 0].mean(dim=(1, 2, 3))
        return features

    def test_scan_catches_synthetic_event_between_baseline_samples(self):
        path, checkpoint = self.video(), self.checkpoint()
        with patch('trace.predict_violence.file_hash', return_value='test-encoder'), \
             patch('trace.predict_violence.load_encoder', return_value=self.fake_encoder):
            baseline = predict(path, checkpoint, 'unused')
            result = scan(path, checkpoint, 'unused')
        self.assertEqual(baseline['prediction'], 'no_violence_flag')
        self.assertEqual(result['prediction'], 'possible_violence')
        self.assertGreater(result['windows_scanned'], 3)
        self.assertTrue(any(segment['start_seconds'] <= 2.5 <= segment['end_seconds']
                            for segment in result['review_segments']))
        self.assertFalse(result['threshold_validated_for_scan'])

    def test_scan_covers_full_timeline_and_preserves_training_preprocessing(self):
        for fps in (15, 29.97):
            with self.subTest(fps=fps):
                path = self.video(fps=fps)
                original, _ = sample_video(path, clips=3)
                windows, first_tensor, last_tensor = [], None, None
                for batch, metadata in iter_video_windows(path):
                    self.assertLessEqual(len(batch), 3)
                    self.assertEqual(tuple(batch.shape[1:]), (3, 16, 112, 112))
                    if first_tensor is None:
                        first_tensor = batch[0]
                    last_tensor = batch[-1]
                    windows.extend(metadata['sampled_windows'])
                self.assertTrue(torch.equal(first_tensor, original[0]))
                self.assertTrue(torch.equal(last_tensor, original[-1]))
                self.assertEqual(windows[0]['start_seconds'], 0)
                self.assertAlmostEqual(windows[-1]['end_seconds'], metadata['duration_seconds'])
                for previous, current in zip(windows, windows[1:]):
                    self.assertLessEqual(current['start_seconds'], previous['end_seconds'])

    def test_short_video_has_one_padded_window(self):
        batches = list(iter_video_windows(self.video(count=4)))
        self.assertEqual(len(batches), 1)
        batch, metadata = batches[0]
        self.assertEqual(len(batch), 1)
        self.assertEqual(metadata['total_windows'], 1)
        self.assertAlmostEqual(metadata['sampled_windows'][0]['end_seconds'], 4 / 15)

    def test_early_close_and_decode_failure_release_video(self):
        with patch('trace.violence.cv2.VideoCapture') as capture_class:
            capture = capture_class.return_value
            capture.get.side_effect = [120, 15]
            capture.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
            with closing(iter_video_windows('unused')) as batches:
                next(batches)
            capture.release.assert_called_once()
        with patch('trace.violence.cv2.VideoCapture') as capture_class:
            capture = capture_class.return_value
            capture.get.side_effect = [120, 15]
            capture.read.return_value = (False, None)
            with self.assertRaisesRegex(ValueError, 'Decode failed'):
                next(iter_video_windows('unused'))
            capture.release.assert_called_once()

    def test_invalid_scan_settings_fail_before_opening_video(self):
        with patch('trace.violence.cv2.VideoCapture') as capture:
            for stride in (0, -1, 1.1, float('nan'), float('inf')):
                with self.subTest(stride=stride), self.assertRaises(ValueError):
                    next(iter_video_windows('unused', stride))
            capture.assert_not_called()
        checkpoint = self.checkpoint()
        with patch('trace.predict_violence.file_hash', return_value='test-encoder'), \
             patch('trace.predict_violence.load_encoder') as loader:
            for threshold in (0, -1, 1, float('nan')):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    scan('unused', checkpoint, 'unused', threshold=threshold)
            loader.assert_not_called()

    def test_review_segments_merge_overlaps_but_preserve_separate_events(self):
        windows = [
            {'start_seconds': 0, 'end_seconds': 1, 'violence_score': .4, 'flagged': True},
            {'start_seconds': .5, 'end_seconds': 1.5, 'violence_score': .7, 'flagged': True},
            {'start_seconds': 1, 'end_seconds': 2, 'violence_score': .1, 'flagged': False},
            {'start_seconds': 3, 'end_seconds': 4, 'violence_score': .5, 'flagged': True},
        ]
        self.assertEqual(merge_review_segments(windows), [
            {'start_seconds': 0, 'end_seconds': 1.5, 'peak_score': .7},
            {'start_seconds': 3, 'end_seconds': 4, 'peak_score': .5},
        ])


if __name__ == '__main__':
    unittest.main()
