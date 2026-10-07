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

## Investigate missed events with a window scan

`run_violence.cmd` now enables experimental **window scan** mode. The original
baseline only samples three roughly one-second windows, then averages their
features into one video score. A brief event between those windows is never
examined, and an event in one sampled window can be diluted by normal footage.

Scan mode checks 16-frame windows at 15 FPS, stepping forward by half a second
throughout the timeline. It includes a final window reaching the last frame,
keeps the original crop/normalization, and scores each window separately. Video
decoding is sequential and image/tensor memory is bounded to a short batch.
The model loads once. Longer videos take more CPU time; progress is printed.

```powershell
# Scan and save per-window scores and approximate review timestamps.
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\Videos\clip.mp4" --scan --output outputs\scan-review.json

# Original whole-video baseline for comparison.
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\Videos\clip.mp4"

# Experiment with a more sensitive threshold on labeled validation footage.
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\Videos\clip.mp4" --scan --threshold 0.2
```

The output path must have an existing parent and must not already exist.
`windows` contains individual scores; `review_segments` merges overlapping
flagged windows. `violence_score` is the maximum window score in scan mode,
not the original whole-video score. `--stride-seconds` accepts values above 0
and at most 1; the default is 0.5. `--threshold` applies only to scan mode.

**This addresses temporal sampling, not proven model accuracy.** The classifier
was trained on averaged whole-video features. Its saved 0.30 threshold has not
been validated for individual windows or a maximum across many windows. Scanning
more windows can increase false alarms; lowering the threshold can increase them
further. The output explicitly marks `threshold_validated_for_scan: false`.
Review intervals are approximate candidate timestamps, not validated event bounds.
The center crop can still omit action near the edges, and the frozen encoder may
not recognize unfamiliar camera views even when the event is sampled.

To assess improvement, collect videos it currently misses with event start/end
times and representative normal recordings. Compare event recall and false alerts
per video/hour on held-out recordings. Tune thresholds on separate validation
recordings; keep recordings from the same source together. If scores remain low
over visible violence, train on representative examples and consider fine-tuning
the video encoder. Repeating the existing head-only training does not adapt that
encoder. The previous 96.31% dataset accuracy does not measure this scan mode or
performance on the user's camera.

## Reproduce on another computer

To run the existing trained model after cloning, follow [FRIEND_SETUP.md](FRIEND_SETUP.md). A clone alone does not include trained model files or the virtual environment.

Use Python 3.12, install `requirements-tested.txt`, and obtain the official R3D-18 weights from https://download.pytorch.org/models/r3d_18-b3b3357e.pth into `models/`. CPU is supported. A CUDA-enabled PyTorch installation is needed for `--device cuda:0`; this laptop's current PyTorch installation is CPU-only.

References: [PyTorch R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html), [RLVS publisher's dataset page](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset). The archive layout matches RLVS, but the archive alone does not establish its provenance or reuse terms.

## Train with another dataset

The current task has two labels: `0 = non_violence` and `1 = violence`. A dataset for the same task can train a new classifier using the same frozen encoder. A dataset labeled stalking, falling, or individual action names needs a new label definition and changes to the classifier, evaluation, and prediction output.

The current trainer always initializes a fresh classifier. It does not resume or fine-tune `best.pt`. To learn from both the original and new datasets, create a combined manifest and retrain, retaining a consistent independent test set. Do not put any existing test videos into training. Review duplicates across both datasets and keep clips from the same original recording together.

If a new ZIP uses the same immediate class-folder names, `Violence` and `NonViolence`, and contains MP4/AVI clips without a prescribed split, the existing importer can prepare it:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_violence.py --archive "C:\path\to\new-dataset.zip" --output data\violence_new
.\.venv\Scripts\python.exe -m trace.train_violence --manifest data\violence_new\manifest.csv --output runs\violence_new --device cpu
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\path\to\clip.mp4" --checkpoint runs\violence_new\best.pt
```

The original baseline is preserved. The launcher still uses the original baseline; use the explicit `--checkpoint` option above to try the new model.

For different layouts, labels, or official splits, write a dataset-specific importer instead of running the RLVS importer unchanged. The trainer accepts a CSV with `path,label,split,sha256,group` columns. Paths are relative to the manifest, `split` is `train`, `val`, or `test`, `sha256` is the video content hash, and `group` identifies the original recording. Include both classes in each split. Preserve official test partitions where supplied. UCF-Crime's long videos and video-level labels need a sampling/labeling strategy appropriate to weak supervision; merely renaming its folders is not sufficient.
