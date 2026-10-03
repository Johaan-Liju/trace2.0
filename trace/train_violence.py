"""Train a reproducible baseline from video features and evaluate a held-out split."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from trace.violence import (RECIPE, binary_metrics, choose_threshold, encode, file_hash,
                            load_encoder, read_manifest)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this PyTorch installation. Use --device cpu.')
    manifest = Path(args.manifest)
    rows = read_manifest(manifest)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Training output already exists. Choose a new --output directory.')
    output.mkdir(parents=True)
    weight_hash = file_hash(args.encoder)
    cache = Path(args.cache) / f'{RECIPE}-{weight_hash[:12]}-clips{args.clips}'
    cache.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.perf_counter()
    usable, excluded, features = [], [], []
    for index, row in enumerate(rows):
        path = row['absolute_path']
        if file_hash(path) != row['sha256']:
            raise ValueError(f'Video content changed since the dataset was prepared: {path}')
        feature_path = cache / (row['sha256'] + '.npy')
        try:
            if feature_path.exists():
                feature = np.load(feature_path, allow_pickle=False)
            else:
                if model is None:
                    model = load_encoder(args.encoder, args.device)
                per_clip, _ = encode(model, path, args.clips, args.device)
                feature = per_clip.mean(dim=0).numpy()
                np.save(feature_path, feature, allow_pickle=False)
            if feature.shape != (512,) or not np.isfinite(feature).all():
                raise ValueError('Invalid cached feature shape or values.')
        except ValueError as exc:
            excluded.append({'path': row['path'], 'split': row['split'], 'label': row['label'], 'reason': str(exc)})
            print(f'EXCLUDED: {row["path"]}: {exc}', flush=True)
            continue
        features.append(feature)
        usable.append(row)
        if (index + 1) % 25 == 0 or index == len(rows) - 1:
            print(f'Features {index + 1}/{len(rows)}; usable={len(usable)}; elapsed={time.perf_counter()-started:.0f}s', flush=True)
    del model
    (output / 'decode_exclusions.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
    x = torch.from_numpy(np.stack(features)).float()
    y = torch.tensor([int(r['label']) for r in usable], dtype=torch.float32)
    masks = {split: torch.tensor([r['split'] == split for r in usable]) for split in ('train', 'val', 'test')}
    for split, mask in masks.items():
        if set(y[mask].tolist()) != {0., 1.}:
            raise ValueError(f'After decode exclusions, {split} does not contain both classes.')
    # Fit normalization on training data only.
    mean = x[masks['train']].mean(0)
    std = x[masks['train']].std(0).clamp_min(.01)
    x = (x - mean) / std
    # Encoder construction consumes RNG state only on cache misses. Reset here
    # so cached and uncached runs initialize the classifier identically.
    torch.manual_seed(42)
    head = torch.nn.Linear(512, 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.05)
    positive = y[masks['train']].sum()
    pos_weight = (masks['train'].sum() - positive) / positive
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss, best_epoch, best_state, stale, history = float('inf'), 0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss = loss_fn(head(x[masks['train']]).squeeze(1), y[masks['train']])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[masks['val']]).squeeze(1), y[masks['val']]).item()
        history.append({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}', flush=True)
        if stale >= 20:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        scores = torch.sigmoid(head(x).squeeze(1)).numpy()
    labels = y.numpy().astype(int)
    threshold = choose_threshold(labels[masks['val']], scores[masks['val']])
    metrics = {split: binary_metrics(labels[mask], scores[mask], threshold) for split, mask in masks.items()}
    checkpoint = {'head': best_state, 'mean': mean, 'std': std, 'threshold': threshold,
                  'clips': args.clips, 'recipe': RECIPE, 'encoder_sha256': weight_hash,
                  'classes': ['non_violence', 'violence'], 'best_epoch': best_epoch}
    torch.save(checkpoint, output / 'best.pt')
    report = {'model': 'Frozen Kinetics-400 R3D-18 + trained linear binary classifier',
              'manifest_sha256': file_hash(manifest), 'encoder_sha256': weight_hash,
              'recipe': RECIPE, 'clips_per_video': args.clips, 'seed': 42,
              'software': {'python': sys.version.split()[0], 'torch': str(torch.__version__)},
              'training': {'optimizer': 'AdamW', 'learning_rate': .01, 'weight_decay': .05,
                           'batch': 'full training split', 'positive_class_weight': float(pos_weight),
                           'early_stopping_patience': 20, 'encoder_frozen': True},
              'best_epoch': best_epoch, 'epochs_run': len(history), 'threshold_selection': 'Maximum validation F1',
              'metrics': metrics, 'decode_exclusions': len(excluded),
              'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['Provisional video-level split; same-source and near-duplicate leakage not ruled out.',
                             'Whole-video labels and sparse sampling: no temporal or live alert evaluation.',
                             'Scores are not calibrated probabilities. This model does not detect stalking.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'source', 'split', 'label', 'violence_score', 'prediction'])
        writer.writeheader()
        for row, score in zip(usable, scores):
            writer.writerow({**{k: row[k] for k in ('path', 'source', 'split', 'label')},
                             'violence_score': float(score), 'prediction': int(score >= threshold)})
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/violence/manifest.csv')
    p.add_argument('--encoder', default='models/r3d_18-b3b3357e.pth')
    p.add_argument('--cache', default='data/violence_features')
    p.add_argument('--output', default='runs/violence_baseline')
    p.add_argument('--clips', type=int, default=3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if min(args.clips, args.epochs, args.threads) < 1:
        p.error('clips, epochs, and threads must be positive.')
    train(args)
