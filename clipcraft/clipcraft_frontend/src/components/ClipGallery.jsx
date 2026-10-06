import { PlayCircle, Download } from 'lucide-react';
import './ClipGallery.css';

export default function ClipGallery({ clips }) {
  if (!clips || clips.length === 0) return null;

  return (
    <div className="clip-gallery animate-fade-in">
      <div className="gallery-header">
        <h3>Extracted Clips</h3>
        <span className="clip-count">{clips.length} results</span>
      </div>
      
      <div className="clips-grid">
        {clips.map(clip => (
          <div key={clip.id} className="clip-card glass-panel" onClick={() => window.open(`http://localhost:8000${clip.videoUrl}`, '_blank')}>
            <div className="clip-thumbnail">
              <video 
                src={`http://localhost:8000${clip.videoUrl}`} 
                className="gallery-video" 
                preload="metadata"
                // No auto-play, just showing the first frame!
                muted 
              />
              <div className="play-overlay">
                <PlayCircle size={40} className="play-icon" />
              </div>
            </div>
            
            <div className="clip-details">
              <h4>{clip.title}</h4>
              <a href={`http://localhost:8000${clip.videoUrl}`} download className="action-btn" title="Download Clip" onClick={(e) => e.stopPropagation()}>
                <Download size={16} />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
