# TRACE violence baseline: 3 October 2026

A binary violence/non-violence classifier has been trained on the supplied `archive (1).zip`. The archive uses the Real Life Violence Dataset folder structure, not the UCF-Crime activity-category structure. This model does not distinguish fighting from other forms of violence and does not detect stalking.

## Dataset

- 4,000 archive entries, including duplicate folder copies.
- 2,014 repeated entries excluded using full-file SHA-256 hashes.
- One distinct video excluded because the same bytes appeared in both classes (`NV_226.mp4` and `V_504.mp4`).
- 1,985 videos retained: 993 non-violence and 992 violence, approximately 2.90 hours.
- Training: 1,389 videos. Validation: 298 videos. Test: 298 videos.
- All retained videos produced sampled features; no videos were excluded at that stage. OpenCV emitted H.264 decoder warnings during the run, so successful sampling should not be taken as a complete bitstream-integrity audit.

The original download was not modified. Extracted videos and the full audit are under `data/violence/`.

## Held-out video results

| Metric | Test result |
| --- | ---: |
| Accuracy | 96.31% |
| Violence precision | 96.00% |
| Violence recall | 96.64% |
| Violence F1 | 96.32% |
| Non-violent clips correctly unflagged | 143 / 149 |
| Non-violent clips incorrectly flagged | 6 / 149 |
| Violent clips correctly flagged | 144 / 149 |
| Violent clips missed | 5 / 149 |

The saved threshold is 0.30, chosen by maximum validation F1. The best checkpoint is epoch 15; early stopping ended training after epoch 35. Validation accuracy is 95.64%. These are whole-video metrics, not event-localization accuracy or false alerts per camera-hour.

## What was trained

The R3D-18 video encoder uses frozen Kinetics-400 pretrained weights. A new linear binary classifier was trained on the averaged embeddings of three sampled windows per video. Training-data statistics normalize the embeddings. Classifier initialization is reset independently of feature caching so cached and uncached runs use the same initialization.

The first feature extraction took approximately 22 minutes on the CPU. Once features were cached, the final classifier-training and evaluation run took about 3.4 seconds, excluding process startup. The preliminary run is retained in `runs/violence_initial_check`; the final reproducible checkpoint and results are in `runs/violence_baseline`.

## Try it

Double-click `run_violence.cmd` and enter a short video's path, or run this from the project folder:

```powershell
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\path\to\clip.mp4"
```

The command prints `possible_violence` or `no_violence_flag`, the model score, and sampled time windows. Scores are not calibrated probabilities. A low score is not proof that a video contains no violence.

The final classifier is `runs/violence_baseline/best.pt`. Keep `models/r3d_18-b3b3357e.pth` alongside it: the classifier depends on that encoder. Full metrics are in `runs/violence_baseline/metrics.json`; per-video scores are in `runs/violence_baseline/predictions.csv`.

## Verification and limitations

All 13 automated tests passed, covering zone events, video sampling, class scores, split leakage checks, duplicate imports, and conflicting labels. Reloading the final checkpoint and decoding one test video from each class reproduced their stored evaluation scores exactly.

This is a provisional video split. Related recordings and differently encoded duplicates have not been manually grouped. Source image sizes are correlated with the labels, so source or scene cues could inflate test performance. New-camera performance is unmeasured. Three short windows can miss brief events, particularly in long recordings. The classifier is not yet connected to the live zone-monitoring window; continuous alerting requires separate evaluation with timed event annotations.

For preparation and retraining commands, see [VIOLENCE_TRAINING.md](VIOLENCE_TRAINING.md).
