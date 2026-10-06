# Run the trained model after cloning

Cloning gets the source code. The repository deliberately excludes `.venv/`, `models/`, `runs/`, and training videos. The message in the screenshot means `runs/violence_baseline/best.pt` is missing. It is not evidence that training failed on the original laptop.

To use the already-trained model, you need two weight files and a local Python environment. You do not need the training dataset or another training run.

## Windows setup

Install Python 3.12 if it is not already installed. Open PowerShell in the cloned repository folder, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
```

Obtain `trace-violence-model.zip` from the person who trained the model. Extract its contents directly into the cloned repository root, alongside `README.md`. For example, if it is in Downloads:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\trace-violence-model.zip" -DestinationPath .
```

Do not extract it into an extra nested folder. The resulting paths must be:

```text
your-clone/
  run_violence.cmd
  .venv/Scripts/python.exe
  models/r3d_18-b3b3357e.pth
  runs/violence_baseline/best.pt
  violence-model-manifest.json
```

Double-click `run_violence.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe -m trace.predict_violence --source "C:\path\to\short-video.mp4"
```

CPU is the default. An NVIDIA GPU is not required for this command. The classifier file is small because the 127 MiB pretrained video encoder is stored separately. The predictor checks that its encoder hash matches the classifier's expected hash.

## Create a shareable model ZIP on the training computer

```powershell
.\.venv\Scripts\python.exe scripts\package_violence.py
```

This creates `dist/trace-violence-model.zip`, verifies the two archived files against SHA-256 hashes, and adds a manifest. It contains no training videos, predictions, or Python environment. Send the ZIP separately to your collaborator. No upload or message is performed by this script. To package again, choose a fresh output filename with `--output`.

The source-code changes in this setup guide and launcher must also be committed and pushed before they appear in another clone. Model files remain separate from ordinary Git history.
