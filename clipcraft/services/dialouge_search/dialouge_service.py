from services.utils.utils import Utils

class DialougeService:

    def __init__(self):
        self.utils = Utils()

    def format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def dialouge_transcribe(self, video_path):
        return self.utils.transcribe_video(video_path)

    def search_transcript(self, transcript: list, query: str) -> list:
        query_lower = query.lower()
        matches = []
        
        for segment in transcript:
            if query_lower in segment["text"].lower():
                matches.append(segment)
                
        return matches
    
    def dialouge_extract_clip(self, video_path, start, end, output_path):
        return self.utils.extract_clip(video_path, start, end, output_path)


