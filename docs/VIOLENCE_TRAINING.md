# Violence video baseline

The trained baseline and measured test results are described in [VIOLENCE_RESULTS.md](VIOLENCE_RESULTS.md).

The supplied `archive (1).zip` has a `Real Life Violence Dataset` folder layout with binary `Violence` and `NonViolence` labels. It is not the UCF-Crime category layout. No metadata, original recording identifiers, temporal event annotations, or dataset license file was included in the archive.

## Dataset audit

The archive contains 4,000 video entries in two copies of the folder structure. SHA-256 deduplication retained 1,986 distinct files. One distinct video appears under both labels (`NV_226.mp4` and `V_504.mp4`); it is excluded from all splits. The importer retains the extracted file for audit but does not include it in training.

The resulting manifest has 1,985 videos: 993 non-violence and 992 violence. Total duration is approximately 2.90 hours.

The median duration is 5 seconds, with a range of 1 to 375.733 seconds. Original image sizes are correlated with the labels: 749 non-violence videos and 199 violence videos are 224 by 224. Resizing both classes for the model does not necessarily remove compression, source, or scene cues. Accuracy on this split may therefore overstate performance on new cameras. Only five videos exceed ten seconds; sparse sampling is particularly limited on these outliers.

| Split | Non-violence | Violence | Total |
| --- | ---: | ---: | ---: |
| Train | 695 | 694 | 1,389 |
| Validation | 149 | 149 | 298 |
| Test | 149 | 149 | 298 |

The split is deterministic and stratified, with seed 42. Exact duplicate files cannot cross splits. However, differently encoded copies, adjacent clips from one recording, and shared scenes have not been manually grouped. These are provisional benchmark splits, not evidence of generalization to new CCTV cameras. The full audit is in `data/violence/dataset_report.json`; individual assignments are in `data/violence/manifest.csv`.

## Model and training

The baseline uses a frozen R3D-18 video encoder pretrained on Kinetics-400, followed by a trained linear binary classifier. It extracts three evenly spaced windows per video, each containing 16 RGB frames sampled at 15 FPS. Each frame is resized to 171 by 128 pixels, center-cropped to 112 by 112, and normalized with the pretrained weight statistics. Window embeddings are averaged into one 512-dimensional video representation.

Only the classifier is trained. Feature normalization uses training data only. Early stopping selects the checkpoint using validation loss. The alert threshold maximizes validation F1 and is then applied unchanged to test videos. The test split does not select the checkpoint or threshold. Features are cached by video checksum, encoder checksum, and sampling recipe so repeat training does not require decoding all videos again.

Video labels apply to the entire recording. Averaging window embeddings provides a simple baseline for these short clips; it does not establish which time interval contains violence. Long videos are sparsely sampled and can contain events between samples. The model cannot label stalking or other individual activity categories.

## Commands

Run from the project root in PowerShell. This environment already has the dataset, dependencies, and encoder weights.

```powershell
# Initial import only. Choose a new output directory to create a different dataset.
.\.venv\Scripts\python.exe scripts\prepare_violence.py --archive "C:\Users\hp\Downloads\archive (1).zip" --output data\violence

# Train. Use a new output directory for subsequent runs.
.\.venv\Scripts\python.exe -m trace.train_violence --output runs\violence_baseline --device cpu

# Classify a short recorded video after training completes.
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\path\to\clip.mp4"

# Train again using cached embeddings, writing to a new run directory.
.\.venv\Scripts\python.exe -m trace.train_violence --output runs\violence_baseline_v2 --device cpu
```

The original `scripts/train.py` is still for YOLO person detection. It must not be used with violence videos.

Training outputs:

- `runs/violence_baseline/best.pt`: classifier, normalization, threshold, and sampling metadata. It requires the separate encoder weights.
- `models/r3d_18-b3b3357e.pth`: frozen video encoder, downloaded from the official PyTorch host and checked against its published hash prefix.
- `runs/violence_baseline/metrics.json`: train/validation/test video metrics and limitations.
- `runs/violence_baseline/predictions.csv`: every accepted video's split, label, score, and prediction.
- `runs/violence_baseline/decode_exclusions.json`: videos excluded if sampled-frame decoding fails.

This is an offline video classification baseline. It is not yet integrated into the YOLO monitoring window. It has no validated event timestamps, live-stream speed, or false-alerts-per-hour measurement. Scores are not calibrated probabilities. Before continuous alerting, evaluate rolling windows on separately annotated, continuous videos and choose alert persistence on validation footage.

## Reproduce on another computer

Use Python 3.12, install `requirements-tested.txt`, and obtain the official R3D-18 weights from https://download.pytorch.org/models/r3d_18-b3b3357e.pth into `models/`. CPU is supported. A CUDA-enabled PyTorch installation is needed for `--device cuda:0`; this laptop's current PyTorch installation is CPU-only.

References: [PyTorch R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html), [RLVS publisher's dataset page](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset). The archive layout matches RLVS, but the archive alone does not establish its provenance or reuse terms.
