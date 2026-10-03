import csv
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from trace.violence import binary_metrics, choose_threshold, read_manifest, sample_video


class ViolenceTests(unittest.TestCase):
    def test_confusion_matrix(self):
        m = binary_metrics([0, 0, 1, 1], [.1, .8, .2, .9], .5)
        self.assertEqual(m['confusion_matrix'], {'true_negative': 1, 'false_positive': 1, 'false_negative': 1, 'true_positive': 1})
        self.assertEqual(m['precision'], .5)
        self.assertEqual(m['recall'], .5)
        self.assertEqual(m['f1'], .5)

    def test_no_positive_predictions_is_defined(self):
        m = binary_metrics([0, 1], [.1, .2], .9)
        self.assertEqual(m['precision'], 0.)
        self.assertEqual(m['recall'], 0.)

    def test_threshold_from_separable_validation(self):
        threshold = choose_threshold([0, 0, 1, 1], [.1, .2, .7, .9])
        self.assertEqual(binary_metrics([0, 0, 1, 1], [.1, .2, .7, .9], threshold)['f1'], 1.)

    def test_cross_split_recording_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'sha256', 'group', 'split', 'label'])
                writer.writeheader()
                writer.writerows([
                    {'path': 'a.mp4', 'sha256': 'a', 'group': 'same-recording', 'split': 'train', 'label': '0'},
                    {'path': 'b.mp4', 'sha256': 'b', 'group': 'same-recording', 'split': 'test', 'label': '1'}])
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                read_manifest(path)

    def test_video_sampler_shape_color_and_short_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'short.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :, 2] = 255  # BGR red; sampler must convert to RGB
                writer.write(frame)
            writer.release()
            tensor, metadata = sample_video(path, clips=3)
            self.assertEqual(tuple(tensor.shape), (3, 3, 16, 112, 112))
            self.assertTrue(tensor.isfinite().all())
            self.assertGreater(tensor[:, 0].mean().item(), 1)
            self.assertLess(tensor[:, 2].mean().item(), 0)
            self.assertEqual(len(metadata['sampled_windows']), 3)
            self.assertLessEqual(metadata['sampled_windows'][0]['end_seconds'], metadata['duration_seconds'])


if __name__ == '__main__':
    unittest.main()
