@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run the setup instructions in README.md first.
  pause
  exit /b 1
)
set /p "TRACE_VIDEO=Enter a video file path or webcam index 0: "
set "TRACE_VIDEO=%TRACE_VIDEO:"=%"
".venv\Scripts\python.exe" -m trace.monitor --source "%TRACE_VIDEO%" --select-zone --save-video
pause
