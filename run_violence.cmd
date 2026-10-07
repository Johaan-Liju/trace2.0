@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Cloning the repository does not copy it.
  echo Follow docs\FRIEND_SETUP.md to create .venv and install the dependencies.
  pause
  exit /b 1
)
if not exist "runs\violence_baseline\best.pt" (
  echo Missing trained classifier: runs\violence_baseline\best.pt
  echo Model weights are excluded from Git and are not included when you clone.
  echo Extract trace-violence-model.zip into this repository folder.
  echo See docs\FRIEND_SETUP.md. You do not need to retrain to run the shared model.
  pause
  exit /b 1
)
if not exist "models\r3d_18-b3b3357e.pth" (
  echo Missing video encoder: models\r3d_18-b3b3357e.pth
  echo Extract the complete trace-violence-model.zip into this repository folder.
  echo See docs\FRIEND_SETUP.md.
  pause
  exit /b 1
)
echo Scanning overlapping windows throughout the video. CPU processing may take time.
echo Flagged timestamps are candidates for review; scan accuracy is not yet validated.
set /p "TRACE_CLIP=Enter the path to a video clip: "
set "TRACE_CLIP=%TRACE_CLIP:"=%"
".venv\Scripts\python.exe" -m trace.predict_violence --source "%TRACE_CLIP%" --scan
pause
