import os
import tempfile
import zipfile
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from services.dialouge_search.dialouge_service import DialougeService
from services.face_detection.face_clipper_service import FaceDetectionService
from services.prompt_search.prompt_search_service import PromptService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure clips directory exists and mount it
os.makedirs("clips", exist_ok=True)
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

dialouge_service = DialougeService()
face_service = FaceDetectionService()
prompt_service = PromptService()


def save_upload_to_temp(upload: UploadFile, suffix: str = "") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or f"_{upload.filename}")
    tmp.write(upload.file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "ClipCraft API"}



@app.post("/api/dialogue-search", summary="Search video dialogue and extract matching clips")
async def dialogue_search(
    video: UploadFile = File(..., description="Video file (mp4, mov, mkv)"),
    query: str = Form(..., description="Text to search for in the transcript"),
):
    video_path = save_upload_to_temp(video, suffix=".mp4")
    print(video_path)
    
    unique_id = int(time.time())

    try:
        segments = dialouge_service.dialouge_transcribe(video_path)
        if not segments:
            raise HTTPException(status_code=422, detail="Transcription produced no segments.")

        matches = dialouge_service.search_transcript(segments, query)
        if not matches:
            raise HTTPException(status_code=404, detail="No matching dialogue found for the given query.")

        clip_paths = []
        for i, match in enumerate(matches):
            filename = f"dialogue_clip_{unique_id}_{i}.mp4"
            out_path = os.path.join("clips", filename)
            result = dialouge_service.dialouge_extract_clip(
                video_path, match["start"], match["end"], out_path
            )
            clip_paths.append(out_path)

        return {"clips": [{"id": i, "title": f"Dialogue Clip {i+1}", "thumbnail": f"/clips/{os.path.basename(path)}", "videoUrl": f"/clips/{os.path.basename(path)}"} for i, path in enumerate(clip_paths)]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face-detection", summary="Detect face scenes and extract clips")
async def face_detection(
    video: UploadFile = File(..., description="Video file (mp4, mov, mkv)"),
    reference_image: Optional[UploadFile] = File(
        None, description="Optional reference face image (jpg, png)"
    ),
):
    video_path = save_upload_to_temp(video, suffix=".mp4")
    ref_path = None
    
    # We will let the face service save to clips natively if possible, or move them.
    # The original extracted to the same folder as the video? Or temp?
    # Actually, face_service.extract_clips returns paths.
    
    try:
        if reference_image is not None:
            ref_path = save_upload_to_temp(reference_image, suffix=".jpg")

        scenes = face_service.detect_scenes(video_path)

        face_scenes = face_service.get_scenes_with_reference(video_path, scenes, ref_path)

        if not face_scenes:
            raise HTTPException(status_code=404, detail="No face scenes found in the video.")

        raw_clip_paths = face_service.extract_clips(video_path, face_scenes)

        if not raw_clip_paths:
            raise HTTPException(status_code=500, detail="Clip extraction failed.")

        unique_id = int(time.time())
        clip_paths = []
        # Move extracted clips to the static clips folder
        for i, path in enumerate(raw_clip_paths):
            new_filename = f"face_clip_{unique_id}_{i}.mp4"
            new_path = os.path.join("clips", new_filename)
            os.rename(path, new_path)
            clip_paths.append(new_path)

        return {"clips": [{"id": i, "title": f"Face Scene {i+1}", "thumbnail": f"/clips/{os.path.basename(path)}", "videoUrl": f"/clips/{os.path.basename(path)}"} for i, path in enumerate(clip_paths)]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/prompt-search", summary="Extract clips matching a text prompt")
async def prompt_search(
    video: UploadFile = File(..., description="Video file (mp4, mov, mkv)"),
    prompt: str = Form(..., description="Describe what you're looking for"),
):
    video_path = save_upload_to_temp(video, suffix=".mp4")

    try:
        best_scenes = prompt_service.run(video_path, prompt)

        if not best_scenes:
            raise HTTPException(status_code=404, detail="No matching scenes found for the prompt.")

        clip_dir = "clips"
        clip_paths = sorted(
            [os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith(".mp4")]
        )
        
        # We only return the latest clips from this prompt search run?
        # PromptService might overwrite or create files in "clips". 
        # We assume they are the ones we want to return.
        if not clip_paths:
            raise HTTPException(status_code=500, detail="Clip extraction produced no files.")

        return {"clips": [{"id": i, "title": f"Prompt Match {i+1}", "thumbnail": f"/{path.replace(os.sep, '/')}", "videoUrl": f"/{path.replace(os.sep, '/')}"} for i, path in enumerate(clip_paths)]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))