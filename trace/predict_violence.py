"""Classify a recorded video using TRACE's trained binary violence baseline."""
import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sys

import torch

from trace.violence import RECIPE, encode, file_hash, iter_video_windows, load_encoder


def predict(source, checkpoint='runs/violence_baseline/best.pt', encoder='models/r3d_18-b3b3357e.pth', device='cpu'):
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['recipe'] != RECIPE or checkpoint['encoder_sha256'] != file_hash(encoder):
        raise ValueError('Encoder or preprocessing differs from the training checkpoint.')
    features, metadata = encode(load_encoder(encoder, device), source, checkpoint['clips'], device)
    head = torch.nn.Linear(512, 1)
    head.load_state_dict(checkpoint['head'])
    head.eval()
    with torch.inference_mode():
        feature = features.mean(0)
        normalized = (feature - checkpoint['mean']) / checkpoint['std']
        score = torch.sigmoid(head(normalized)).item()
    return {'video': str(source), 'prediction': 'possible_violence' if score >= checkpoint['threshold'] else 'no_violence_flag',
            'violence_score': score, 'threshold': checkpoint['threshold'], **metadata,
            'note': 'Video-level prototype score, not a calibrated probability. Sparse samples can miss events; no stalking classification.'}


def merge_review_segments(windows):
    """Combine touching flagged windows into approximate intervals for review."""
    segments = []
    for window in windows:
        if not window['flagged']:
            continue
        if segments and window['start_seconds'] <= segments[-1]['end_seconds'] + 1e-6:
            segments[-1]['end_seconds'] = max(segments[-1]['end_seconds'], window['end_seconds'])
            segments[-1]['peak_score'] = max(segments[-1]['peak_score'], window['violence_score'])
        else:
            segments.append({'start_seconds': window['start_seconds'], 'end_seconds': window['end_seconds'],
                             'peak_score': window['violence_score']})
    return segments


def scan(source, checkpoint='runs/violence_baseline/best.pt', encoder='models/r3d_18-b3b3357e.pth',
         device='cpu', stride_seconds=0.5, threshold=None, progress=None):
    """Score overlapping windows instead of averaging a few video-wide samples.

    This is an experimental use of a video-trained head: its window threshold
    needs validation on timed, representative footage before automated alerts.
    """
    if not math.isfinite(stride_seconds) or not 0 < stride_seconds <= 1:
        raise ValueError('Scan stride must be greater than 0 and at most 1 second.')
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['recipe'] != RECIPE or checkpoint['encoder_sha256'] != file_hash(encoder):
        raise ValueError('Encoder or preprocessing differs from the training checkpoint.')
    chosen_threshold = checkpoint['threshold'] if threshold is None else threshold
    if not math.isfinite(chosen_threshold) or not 0 < chosen_threshold < 1:
        raise ValueError('Threshold must be greater than 0 and less than 1.')
    model = load_encoder(encoder, device)
    head = torch.nn.Linear(512, 1)
    head.load_state_dict(checkpoint['head'])
    head.eval()
    windows = []
    with closing(iter_video_windows(source, stride_seconds)) as batches, torch.inference_mode():
        for batch, metadata in batches:
            features = model(batch.to(device)).cpu()
            normalized = (features - checkpoint['mean']) / checkpoint['std']
            scores = torch.sigmoid(head(normalized)).flatten().tolist()
            for window, score in zip(metadata['sampled_windows'], scores):
                windows.append({**window, 'violence_score': score, 'flagged': score >= chosen_threshold})
            if progress:
                progress(len(windows), metadata['total_windows'])
    peak = max(window['violence_score'] for window in windows)
    return {'video': str(source), 'mode': 'window_scan',
            'prediction': 'possible_violence' if peak >= chosen_threshold else 'no_violence_flag',
            'violence_score': peak, 'score_aggregation': 'maximum_window_score',
            'threshold': chosen_threshold, 'threshold_validated_for_scan': False,
            'duration_seconds': metadata['duration_seconds'], 'stride_seconds': stride_seconds,
            'windows_scanned': len(windows), 'review_segments': merge_review_segments(windows),
            'windows': windows,
            'note': 'Experimental window scores, not calibrated probabilities. Review flagged timestamps; '
                    'the threshold was not validated for individual windows. More windows can increase false alarms. '
                    'No flag does not establish absence of violence.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', default='runs/violence_baseline/best.pt')
    p.add_argument('--encoder', default='models/r3d_18-b3b3357e.pth')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', help='Optional new JSON output path')
    p.add_argument('--scan', action='store_true', help='Check overlapping windows throughout the video and report timestamps')
    p.add_argument('--stride-seconds', type=float, default=0.5, help='Scan window step: greater than 0 and at most 1 (default: 0.5)')
    p.add_argument('--threshold', type=float, help='Experimental scan threshold; lower values flag more windows')
    args = p.parse_args()
    if not math.isfinite(args.stride_seconds) or not 0 < args.stride_seconds <= 1:
        p.error('--stride-seconds must be greater than 0 and at most 1.')
    if args.threshold is not None and (not math.isfinite(args.threshold) or not 0 < args.threshold < 1):
        p.error('--threshold must be greater than 0 and less than 1.')
    if not args.scan and (args.threshold is not None or args.stride_seconds != 0.5):
        p.error('--threshold and --stride-seconds require --scan.')
    torch.set_num_threads(8)
    if args.scan:
        def report_progress(done, total):
            if done == 3 or done % 30 == 0 or done == total:
                print(f'Scanned {done}/{total} windows', file=sys.stderr, flush=True)
        result = scan(args.source, args.checkpoint, args.encoder, args.device,
                      args.stride_seconds, args.threshold, report_progress)
    else:
        result = predict(args.source, args.checkpoint, args.encoder, args.device)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
