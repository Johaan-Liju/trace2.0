# ClipCraft

ClipCraft is an intelligent, open-source video clipper. It enables users to automatically extract specific scenes and segments from video files using advanced techniques like dialogue transcription, facial detection, and semantic text prompts.

## Features

- **Dialogue Search**: Transcribes video audio and extracts exact clips matching specific dialogue queries.
- **Face Detection**: Automatically detects scenes containing faces, with the ability to isolate scenes matching a specific reference image.
- **Prompt Search**: Interprets natural language prompts to locate and clip visually matching scenes from a video.

## Project Structure

- **Backend**: A FastAPI-based REST API (`server.py`) handles video processing, model inference, and clip extraction. An alternative standalone Streamlit testing interface is also available (`main.py`).
- **Frontend**: A modern React application powered by Vite (`clipcraft_frontend`) that provides a user interface for interacting with the backend services.

## Prerequisites

Before building the project, ensure you have the following installed:

- **Python 3.9+**
- **Node.js 18+** and **npm**
- **FFmpeg**: Must be installed and accessible via your system's PATH. This is required for video clipping and processing.
- **Ollama**: Must be installed and accessible via your system's PATH. This is required for running the promnt features.

## Setup & Build Instructions

### 1. Backend Setup

First, set up the Python backend to handle video processing.

```bash
# Navigate to the project root directory
cd clipcraft

# (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required Python dependencies
pip install -r requirements.txt
```

You can start the backend services in one of two ways:

**Option A: FastAPI Backend (Recommended)**
Run the REST API designed to communicate with the React frontend. By default, it will run on `http://127.0.0.1:8000`.
```bash
uvicorn server:app --reload
```

**Option B: Streamlit Interface**
Run the standalone Streamlit desktop interface.
```bash
streamlit run main.py
```

### 2. Frontend Setup

Next, set up the React client to interact with the FastAPI server.

```bash
# Navigate to the frontend directory
cd clipcraft_frontend

# Install Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```

## API Reference

When running the FastAPI server, the following core REST endpoints are available:

- `POST /api/dialogue-search`: Upload a video and provide a text query to extract clips with matching dialogue.
- `POST /api/face-detection`: Upload a video and an optional reference image to extract clips containing specific faces.
- `POST /api/prompt-search`: Upload a video and provide a descriptive prompt to isolate relevant visual events.
