import { UploadCloud, Play, Pause, Maximize, Volume2 } from 'lucide-react';
import { useRef, useState } from 'react';
import './VideoPlayer.css';

export default function VideoPlayer({ videoFile, onUpload }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef(null);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      onUpload(file);
    }
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  if (!videoFile) {
    return (
      <div 
        className="upload-container glass-panel" 
        onDragOver={(e) => e.preventDefault()} 
        onDrop={handleDrop}
      >
        <div className="upload-content">
          <div className="upload-icon-wrapper">
            <UploadCloud size={48} className="upload-icon" />
          </div>
          <h2>Upload Video</h2>
          <p>Drag and drop your video file here, or click to browse.</p>
          <label className="btn-primary upload-btn">
            Browse Files
            <input 
              type="file" 
              accept="video/*" 
              className="hidden-input" 
              onChange={handleChange} 
            />
          </label>
        </div>
      </div>
    );
  }

  const videoUrl = URL.createObjectURL(videoFile);

  return (
    <div className="video-player-container glass-panel animate-fade-in">
      <div className="video-wrapper">
        <video 
          ref={videoRef}
          src={videoUrl} 
          className="video-element"
          onEnded={() => setIsPlaying(false)}
        />
        
        <div className="custom-controls">
          <button className="control-btn" onClick={togglePlay}>
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          
          <div className="progress-bar">
            <div className="progress-filled" style={{ width: '0%' }}></div>
          </div>
          
          <div className="control-group">
            <button className="control-btn"><Volume2 size={20} /></button>
            <button className="control-btn"><Maximize size={20} /></button>
          </div>
        </div>
      </div>
      <div className="video-info">
        <h3>{videoFile.name}</h3>
        <p className="file-size">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</p>
      </div>
    </div>
  );
}
