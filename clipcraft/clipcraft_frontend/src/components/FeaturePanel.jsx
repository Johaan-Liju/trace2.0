import { useState, useRef } from 'react';
import { Search, ScanFace, Wand2, Sparkles, Filter, Upload, Image as ImageIcon } from 'lucide-react';
import './FeaturePanel.css';

export default function FeaturePanel({ activeFeature, onProcess, hasVideo }) {
  const [query, setQuery] = useState('');
  const [referenceImage, setReferenceImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setReferenceImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleAction = (e) => {
    e.preventDefault();
    if (!hasVideo) {
      alert("Please upload a video first!");
      return;
    }
    
    if (activeFeature === 'face' && !referenceImage) {
      alert("Please upload a reference image for face detection!");
      return;
    }

    onProcess({ feature: activeFeature, query, referenceImage });
  };

  const renderContent = () => {
    switch (activeFeature) {
      case 'search':
        return (
          <div className="feature-content animate-fade-in">
            <div className="icon-header">
              <div className="icon-wrapper bg-blue"><Search size={28} /></div>
              <h2>Dialogue Search</h2>
              <p>Find specific spoken words or phrases in the video transcript.</p>
            </div>
            <form onSubmit={handleAction} className="action-form">
              <div className="input-group">
                <label>Search Query</label>
                <div className="input-with-icon">
                  <Search size={18} className="input-icon" />
                  <input 
                    type="text" 
                    className="input-base" 
                    placeholder="e.g. 'I'll be back'..." 
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
              </div>
              <button type="submit" className="btn-primary full-width">
                Search Transcript
              </button>
            </form>
          </div>
        );

      case 'face':
        return (
          <div className="feature-content animate-fade-in">
             <div className="icon-header">
              <div className="icon-wrapper bg-green"><ScanFace size={28} /></div>
              <h2>Face Detection</h2>
              <p>Detect and isolate clips featuring specific people by uploading a reference image.</p>
            </div>
            <form onSubmit={handleAction} className="action-form">
              <div className="input-group">
                <label>Reference Image</label>
                <div 
                  className="image-upload-area" 
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input 
                    ref={fileInputRef}
                    type="file" 
                    accept="image/*"
                    onChange={handleImageChange}
                    style={{ display: 'none' }}
                  />
                  {imagePreview ? (
                    <div className="image-preview-container">
                      <img src={imagePreview} alt="Reference Preview" className="reference-preview" />
                      <div className="change-image-overlay">
                        <Upload size={16} /> Change Image
                      </div>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <ImageIcon size={24} className="upload-icon" />
                      <span>Click to upload a face</span>
                      <span className="upload-hint">JPG, PNG up to 5MB</span>
                    </div>
                  )}
                </div>
              </div>
              <button type="submit" className="btn-primary full-width">
                Scan for Faces
              </button>
            </form>
          </div>
        );

      case 'prompt':
        return (
          <div className="feature-content animate-fade-in">
            <div className="icon-header">
              <div className="icon-wrapper bg-purple"><Wand2 size={28} /></div>
              <h2>Prompt Clipping</h2>
              <p>Use AI to semantically extract clips matching your description.</p>
            </div>
            <form onSubmit={handleAction} className="action-form">
              <div className="input-group">
                <label>Semantic Prompt</label>
                <textarea 
                  className="input-base textarea" 
                  placeholder="e.g. 'An explosion happens in the background'..." 
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={4}
                />
              </div>
              <button type="submit" className="btn-primary full-width">
                <Sparkles size={16} /> Generate Magic Clips
              </button>
            </form>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="feature-panel glass-panel">
      {renderContent()}
    </div>
  );
}
