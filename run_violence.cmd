@echo off
cd /d "%~dp0"
if not exist "runs\violence_baseline\best.pt" (
  echo The violence classifier is not ready. See docs\VIOLENCE_TRAINING.md.
  pause
  exit /b 1
)
set /p "TRACE_CLIP=Enter the path to a short video clip: "
set "TRACE_CLIP=%TRACE_CLIP:"=%"
".venv\Scripts\python.exe" -m trace.predict_violence --source "%TRACE_CLIP%"
pause
