"""Shared sampling, frozen R3D-18 features, and binary violence classifier."""
import csv
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import r3d_18

RECIPE = 'r3d18-rgb-resize171x128-center112-16frames15fps-v1'


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    owners = {}
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'} or row['label'] not in {'0', '1'}:
            raise ValueError('Manifest has invalid split or class label.')
        for key in (row['sha256'], row['group']):
            if key in owners and owners[key] != row['split']:
                raise ValueError('A video hash or recording group occurs in multiple splits.')
            owners[key] = row['split']
        row['absolute_path'] = str((path.parent / row['path']).resolve())
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {'0', '1'}:
            raise ValueError(f'{split} needs both classes.')
    return rows


def sample_video(path, clips=3):
    """Sample 16 frames at 15 FPS from evenly spaced ~1 second windows.

    Whole-video labels supervise the mean embedding of all sampled windows.
    Clips shorter than a window use repeated boundary frames. Long videos are
    sparsely sampled and may contain events between samples.
    """
    if clips < 1:
        raise ValueError('clips must be positive')
    cap = cv2.VideoCapture(str(path))
    try:
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or count < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Unreadable video metadata: {path}')
        span = 15 * fps / 15.0
        last_start = max(0, count - 1 - span)
        starts = np.linspace(0, last_start, clips) if clips > 1 else np.array([last_start / 2])
        indices = np.rint(starts[:, None] + np.arange(16)[None, :] * fps / 15).astype(int)
        indices = np.clip(indices, 0, count - 1)
        wanted, decoded = set(indices.flatten().tolist()), {}
        # One sequential pass for short dataset clips avoids repeated expensive seeks.
        if count <= 2000:
            for index in range(int(indices.max()) + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Decode failed at frame {index}: {path}')
                if index in wanted:
                    decoded[index] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (171, 128))
        else:
            for index in sorted(wanted):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Decode failed at sampled frame {index}: {path}')
                decoded[index] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (171, 128))
        frames = np.stack([[decoded[int(i)] for i in window] for window in indices])
        tensor = torch.from_numpy(frames).permute(0, 4, 1, 2, 3).float().div_(255)
        # Torchvision center crop uses round((size-crop)/2).
        tensor = tensor[:, :, :, 8:120, 30:142]
        mean = torch.tensor([.43216, .394666, .37645]).view(1, 3, 1, 1, 1)
        std = torch.tensor([.22803, .22145, .216989]).view(1, 3, 1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.contiguous(), {'duration_seconds': count / fps,
                'sampled_windows': [{'start_seconds': float(a[0] / fps), 'end_seconds': float((a[-1] + 1) / fps)} for a in indices]}
    finally:
        cap.release()


def load_encoder(weights, device='cpu'):
    model = r3d_18(weights=None)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    return model.eval().to(device)


@torch.inference_mode()
def encode(model, path, clips, device):
    batch, metadata = sample_video(path, clips)
    # Keep memory bounded even if the user increases the number of sampled clips.
    features = torch.cat([model(chunk.to(device)).cpu() for chunk in batch.split(3)])
    return features, metadata


def binary_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores)
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {'videos': len(labels), 'threshold': float(threshold), 'accuracy': (tp + tn) / max(1, len(labels)),
            'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / max(1e-12, precision + recall),
            'confusion_matrix': {'true_negative': tn, 'false_positive': fp, 'false_negative': fn, 'true_positive': tp}}


def choose_threshold(labels, scores):
    # Chosen on validation only; never tune this on test predictions.
    candidates = [binary_metrics(labels, scores, t) for t in np.linspace(.05, .95, 181)]
    return max(candidates, key=lambda m: (m['f1'], m['precision'], -abs(m['threshold'] - .5)))['threshold']
