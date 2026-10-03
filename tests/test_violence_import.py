from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import cv2
import numpy as np

from scripts.prepare_violence import prepare


class ImportTests(unittest.TestCase):
    def test_duplicate_archive_and_conflicting_label_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'input.zip'
            contents = []
            for index in range(24):
                video = root / f'{index}.avi'
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 15, (32, 32))
                self.assertTrue(writer.isOpened())
                frame = np.full((32, 32, 3), index * 10, dtype=np.uint8)
                for _ in range(2):
                    writer.write(frame)
                writer.release()
                contents.append(video.read_bytes())
            with ZipFile(archive, 'w') as z:
                for index, content in enumerate(contents):
                    label = 'NonViolence' if index < 12 else 'Violence'
                    z.writestr(f'first/{label}/{index}.avi', content)
                    z.writestr(f'duplicate/{label}/{index}.avi', content)
                z.writestr('conflict/Violence/conflict.avi', contents[0])
            with redirect_stdout(io.StringIO()):
                prepare(archive, root / 'prepared')
            report = json.loads((root / 'prepared/dataset_report.json').read_text())
            self.assertEqual(report['unique_readable_videos'], 23)
            self.assertEqual(report['conflicting_label_videos_excluded'], 1)
            self.assertEqual(report['duplicates_removed'], 25)
            with (root / 'prepared/manifest.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len({r['sha256'] for r in rows}), 23)
            self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
            self.assertFalse({r['sha256'] for r in rows} & {r['sha256'] for r in report['rejected']})
            with self.assertRaises(ValueError):
                prepare(archive, root / 'prepared', resume=True)


if __name__ == '__main__':
    unittest.main()
