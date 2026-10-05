import tempfile
import ffmpeg
import whisper
import os
import subprocess
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import yt_dlp

class Utils:
    def __init__(self):
        pass

    def detect_scenes(self, video):
        video_stream = open_video(video)
        
        # Get FPS to calculate frame count if you want to work in seconds
        fps = video_stream.frame_rate 
        min_duration_seconds = 3
        min_frames = int(min_duration_seconds * fps)

        scene_manager = SceneManager()
        
        # Add the parameter here
        scene_manager.add_detector(ContentDetector(min_scene_len=min_frames))
        
        scene_manager.detect_scenes(video_stream)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for start, end in scene_list:
            scenes.append((start.get_seconds(), end.get_seconds()))

        print("Scenes:", scenes)
        return scenes

    def transcribe_video(self, video_path) -> list:   
        """
        "text": segment["text"].strip(),
        "start": segment["start"],
        "end": segment["end"]
        """          
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        segments = []
        try:
            ffmpeg.input(video_path).output(temp_audio_path, ac=1, ar=16000, loglevel="error").overwrite_output().run()
            
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio_path)
            
            for segment in result["segments"]:
                segments.append({
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
        return segments

    def extract_clip(self, video_path, start, end, output_path) -> str:
         # Need to make changes such that it can take a list of timestamps

        start_time = max(0, start - 0.5)
        end_time = end + 0.5
        duration = end_time - start_time
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        (
            ffmpeg
            .input(video_path, ss=start_time)
            .output(output_path, t=duration, loglevel="error")
            .overwrite_output()
            .run()
        )
        return output_path
    """
    ALTERNATE EXTRACT CLIPS FUNCTION
    """
    def extract_clips(self, video, best_scenes):
        os.makedirs("clips", exist_ok=True)
        extracted_files = []

        for i, scene in enumerate(best_scenes, start=1):
            start, end = scene
            output_file = f"clips/clip_{i}.mp4"

            command = [
                "ffmpeg",
                "-i", video,
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                "-y",  # Automatically overwrite output files
                output_file
            ]

            subprocess.run(command, check=True)
            extracted_files.append(output_file)

        print("Clips extracted successfully.")
        return extracted_files
    