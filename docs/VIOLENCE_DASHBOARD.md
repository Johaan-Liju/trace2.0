# Camera and violence review in one dashboard

Start from the repository root:

```powershell
.\.venv\Scripts\python.exe dashboard.py
```

Open **http://127.0.0.1:8765**. Restart the server after updating its Python code
and refresh the browser to load the new interface. Both existing weight files
are required: `runs/violence_baseline/best.pt` and
`models/r3d_18-b3b3357e.pth`. The camera can still run if they are missing; the
Violence review page identifies missing files.

## Analyse an existing video

1. Open **Violence review** in the sidebar.
2. Choose an MP4, AVI, MOV, MKV, or WebM file up to 256 MB.
3. Select **Upload & analyse**. The file is copied to this computer, not uploaded
   to a cloud service.
4. Follow progress or cancel the job. Scans run one at a time; camera controls
   remain available. CPU contention can reduce camera FPS while scanning.
5. Select a completed analysis. Review its peak score, score timeline, and
   candidate timestamps. Click a timestamp or the chart to seek the video.
6. Download the original video or the JSON report with all per-window scores.

Browser playback depends on the video codec. H.264 MP4 is a suitable format.
If the browser cannot decode a file that the model can read, download the video
and review it in a local video player.

## Connect camera recordings automatically

1. Stop the camera and open **Settings**.
2. Enable **Record entry clips** and **Automatically analyse entry clips**, then
   save. FFmpeg must be available for clip recording.
3. Start **AI monitoring**. A person entering the camera view starts an entry
   recording using the existing clip settings.
4. Once the MP4 finishes, it is queued for a violence scan. Results appear under
   **Violence review**, linked to that recording.

You can also select **Analyse violence** on any recording from this session.
Re-selecting a queued, running, or completed recording opens its existing scan;
cancelled or failed scans can be retried. Automatic scan errors appear beside the
recording and do not interrupt camera processing.

This checks **recorded entry clips only**, after encoding finishes. It does not
cover every moment of a continuous feed. A person remaining in view does not
trigger repeated entry clips, so later activity may not be recorded or analysed.
Use an uploaded full recording when you need to scan its complete timeline.
Zone-entry alerts and their existing sound notifications are independent of the
violence classifier; a candidate violence result does not produce a new alarm.

## What the scores mean

The scan uses overlapping windows with the classifier's saved threshold. Scores
are not probabilities; flagged intervals are candidates for human review. The
threshold has not been validated for window scanning, and the earlier dataset
accuracy is not a measurement of this integrated workflow. No flag does not
establish that a recording contains no violence. The frontend integration does
not retrain the model or resolve its recognition limitations.

## Storage and lifecycle

- Uploaded copies, reports, and diagnostic logs are under `storage/violence/`.
  Recordings remain in the camera's configured clips folder.
- The interface lists the latest 100 analyses from this server session. Files
  remain after a restart, but the in-memory index is not restored. Download
  reports you need, or open their JSON files directly from storage.
- Up to 16 scans may be queued/running. If automatic scanning cannot enqueue a
  clip, its recording row reports the error so you can retry later.
- Closing the browser does not stop jobs. Cancel a job in the UI to stop it.
  Stopping the dashboard server cancels pending/running scans and reaps the
  active scanner process.
- This remains a loopback-only dashboard without remote accounts or hosting.

## Code flow

```text
camera -> completed entry MP4 -----+
recording's Analyse button -------+--> ViolenceScans queue
browser upload -> local copy -----+          |
                                            v
                              trace.predict_violence --scan
                                            |
                               progress log + JSON report
                                            |
                              browser video/timeline review
```

`services/violence_service.py` owns the bounded queue and scanner subprocess.
`services/web_service.py` provides authenticated local POST routes, upload
handling, registered video/report downloads, and byte ranges for seeking.
`frontend/violence.js` renders progress and review results; `frontend/app.js`
connects the camera settings and recordings page. The CLI classifier remains
usable independently.
