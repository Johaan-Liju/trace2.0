from services.prompt_search.prompt_clip import run_prompt_clipper


class PromptService:
    health_string = "Prompt Service is UP"

    def __init__(self):
        pass

    def health(self):
        return self.health_string

    def run(self, video_path, prompt):
        return run_prompt_clipper(video_path, prompt)