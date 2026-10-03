"""Classify a recorded video using TRACE's trained binary violence baseline."""
import argparse
import json
from pathlib import Path

import torch

from trace.violence import RECIPE, encode, file_hash, load_encoder


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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', default='runs/violence_baseline/best.pt')
    p.add_argument('--encoder', default='models/r3d_18-b3b3357e.pth')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', help='Optional new JSON output path')
    args = p.parse_args()
    torch.set_num_threads(8)
    result = predict(args.source, args.checkpoint, args.encoder, args.device)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
