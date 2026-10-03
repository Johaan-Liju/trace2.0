"""Fine-tune a pretrained person detector, or verify training with a tiny sample."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Dataset YAML from prepare_dataset.py')
    p.add_argument('--model', default='models/yolo11n.pt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--name', default='person_baseline')
    p.add_argument('--smoke', action='store_true', help='1 epoch at 320px; verifies plumbing only')
    p.add_argument('--evaluate-only', choices=['val', 'test'])
    args = p.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        p.error('Epochs, image size, and batch must be positive.')
    config_dir = Path('.runtime/ultralytics').resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(config_dir))
    import yaml
    from ultralytics import YOLO
    data = Path(args.data).resolve()
    if not data.is_file():
        p.error('Dataset YAML does not exist. Follow docs/TRAINING.md.')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    names = config.get('names')
    if not args.smoke and names not in ({0: 'person'}, ['person']):
        p.error('Use a one-class dataset with names: {0: person}. --smoke permits sample datasets.')
    if args.evaluate_only and args.evaluate_only not in config:
        p.error(f'Dataset has no {args.evaluate_only} split.')
    model = YOLO(args.model)
    common = dict(data=str(data), imgsz=320 if args.smoke else args.imgsz,
                  device=args.device, batch=2 if args.smoke else args.batch,
                  workers=0, project='runs', name=args.name, exist_ok=False)
    if args.evaluate_only:
        metrics = model.val(split=args.evaluate_only, **common)
    else:
        model.train(epochs=1 if args.smoke else args.epochs, seed=42,
                    deterministic=True, patience=8, amp=False if args.device == 'cpu' else True,
                    plots=True, **common)
        best = Path(model.trainer.best)
        print(f'Best checkpoint: {best}')
        metrics = YOLO(str(best)).val(**{**common, 'name': args.name + '_validation'})
    summary = {'smoke_test_only': args.smoke, 'evaluation_split': args.evaluate_only or 'val',
               'metrics': {k: float(v) for k, v in metrics.results_dict.items()}}
    Path(metrics.save_dir, 'trace_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if args.smoke:
        print('SMOKE TEST ONLY: these metrics do not establish CCTV performance.')


if __name__ == '__main__':
    main()

