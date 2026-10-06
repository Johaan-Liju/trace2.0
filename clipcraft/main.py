import os
import tempfile
import streamlit as st

from services.dialouge_search.dialouge_service import DialougeService
from services.face_detection.face_clipper_service import FaceDetectionService
from services.prompt_search.prompt_search_service import PromptService

dialouge_service = DialougeService()
face_service = FaceDetectionService()
prompt_service = PromptService()

def face_clipper():
    st.header("Face Scene Detection")

    #1) Upload files
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "mkv"])
    uploaded_ref = st.file_uploader("Upload reference image (optional)", type=["jpg", "jpeg", "png"])

    if uploaded_video:
        st.session_state["video_path"] = save_uploaded_file(uploaded_video)
        st.success(f"Video uploaded: {st.session_state['video_path']}")

    if uploaded_ref:
        st.session_state["ref_path"] = save_uploaded_file(uploaded_ref)
        print(st.session_state["ref_path"])
        st.success(f"Reference image uploaded: {st.session_state['ref_path']}")

    video_path = st.session_state.get("video_path")
    ref_path = st.session_state.get("ref_path")

    # 2) Detect scenes
    if video_path and st.button("Detect Scenes"):
        with st.spinner("Detecting scenes..."):
            scenes = face_service.detect_scenes(video_path)
            st.session_state["scenes"] = scenes
            st.success(f"{len(scenes)} scenes detected")
            st.write(scenes)

    # 3) Face scene selection
    scenes = st.session_state.get("scenes", [])
    if scenes:
        st.markdown("### Face search mode")
        mode = st.radio("Choose mode", ["Any face", "Reference face"], index=0)
        face_scenes = []
        if st.button("Find Face Scenes"):
            with st.spinner("Searching face scenes..."):
                if mode == "Reference face" and ref_path:
                    face_scenes = face_service.get_scenes_with_reference(video_path, scenes, ref_path)
                    #To be implemented
                else:
                    face_scenes = face_service.get_face_scenes(video_path, scenes)
                st.session_state["face_scenes"] = face_scenes
                st.success(f"Found {len(face_scenes)} face scenes")
                st.write(face_scenes)

    # 4) Clip extraction
    face_scenes = st.session_state.get("face_scenes", [])
    if face_scenes:
        st.subheader("Clip extraction")
        if st.button("Extract Clips"):
            with st.spinner("Extracting clips..."):
                result_paths = face_service.extract_clips(video_path, face_scenes)
                st.session_state["extracted_clips"] = result_paths
                st.success(f"Extracted {len(result_paths)} clips successfully!")

    # Display extracted clips
    extracted_clips = st.session_state.get("extracted_clips", [])
    if extracted_clips:
        st.subheader("🎬 Extracted Clips")
        for i, clip_path in enumerate(extracted_clips, 1):
            st.markdown(f"**Clip {i}**: `{clip_path}`")
            if os.path.exists(clip_path):
                st.video(clip_path)
            else:
                st.warning(f"File not found: {clip_path}")

def video_prompt():
    st.title("Prompt-Based Clip Extractor")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    prompt = st.text_input("Enter prompt")

    if st.button("Extract Clips"):
        if uploaded_video is None:
            st.error("Upload a video first")
            return
        if not prompt:
            st.error("Enter a prompt")
            return

        with open(uploaded_video.name, "wb") as f:
            f.write(uploaded_video.getbuffer())

        with st.spinner("Processing video..."):
            result = prompt_service.run(uploaded_video.name, prompt)

        st.success("Clips extracted successfully!")

        import subprocess
        import os

        st.subheader("Extracted Clips")

        for i, seg in enumerate(result):
            start = seg[0]
            end = seg[1]

            st.write(f"{i}: Start = {start}s, End = {end}s")

            output_file = f"clip_{i}.mp4"

            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", uploaded_video.name,
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                output_file
            ])

            st.video(output_file)
        






def dialouge_search():
    st.header("Dialogue Search")

    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "mkv"])
    if uploaded_video:
        video_path = save_uploaded_file(uploaded_video)
        st.session_state["video_path"] = video_path
        st.success(f"File saved: {video_path}")

        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                segments = dialouge_service.dialouge_transcribe(video_path)
                st.session_state["transcript_segments"] = segments
                st.write("Transcription Completed")

    segments = st.session_state.get("transcript_segments", [])
    video_path = st.session_state.get("video_path", None)
    query = st.text_input("Search transcript for", "")

    if st.button("Search"):
        if not segments:
            st.warning("Transcribe first.")
        else:
            matches = dialouge_service.search_transcript(segments, query)
            st.session_state["matched_segments"] = matches
            st.write("Matches:", matches)

    matched_segments = st.session_state.get("matched_segments", [])
    
    if matched_segments and video_path:
        st.subheader("Clip extraction")
        selected_idx = st.selectbox("Select a match to extract", range(len(matched_segments)), 
                                     format_func=lambda i: f"{i}: {matched_segments[i]['text'][:50]}")
        selected_match = matched_segments[selected_idx]
        start = selected_match["start"]
        end = selected_match["end"]
        
        st.write(f"Selected: {selected_match['text']}")
        st.write(f"Start: {start}s, End: {end}s")
        
        out_name = st.text_input("Output filename", "clip.mp4")
        if st.button("Extract Clip"):
            with st.spinner("Extracting clip..."):
                out_path = os.path.join(tempfile.gettempdir(), out_name)
                result = dialouge_service.dialouge_extract_clip(video_path, start, end, out_path)
                st.success(f"Saved clip: {result}")
                st.video(result)

def save_uploaded_file(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def main():
    st.title("ClipCraft")
    pg = st.navigation([
        st.Page(dialouge_search, title="Dialogue Search"),
        st.Page(face_clipper, title="Face Detection"), 
        st.Page(video_prompt, title="Prompt Search")
    ])
    pg.run()

if __name__ == "__main__":
    main()
