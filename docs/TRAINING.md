# Training recipe

1. Use the pretrained `models/yolo11n.pt` checkpoint to establish a no-fine-tuning baseline.
2. Extract frames from each camera. Split by recording session, not by neighboring frames. Never let frames from one continuous clip appear in both train and test.
3. Label the `person` class only. Include partially visible people if the detector should support them. Keep reviewed empty frames with empty `.txt` labels.
4. Run `scripts/prepare_dataset.py`. It requires explicit train, val, and test rows and checks the YOLO normalized box format.
5. Run the one-epoch smoke test, then a short CPU run. Compare the test metrics with the pretrained baseline before adding more data.
6. Use the trained checkpoint with `trace.monitor --model runs/<name>/weights/best.pt`. Keep the restricted-zone rule outside the model so its policy can change without retraining.
7. Tune `--dwell` and `--missing-timeout` against reviewed clips. Record event precision, event recall, median alert delay, and false alerts per camera-hour.

The monitor is a behavior-rule prototype, not a complete anomaly detector. Fighting, falls, abandoned objects, and unusual behavior need separate temporal datasets and evaluation protocols. UCF-Crime is a research starting point for coarse anomaly scoring, but its weak video-level labels are not enough for reliable per-frame alerts without additional annotation.

