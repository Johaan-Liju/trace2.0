import { useState } from 'react'
import Sidebar from './components/Sidebar'
import FeaturePanel from './components/FeaturePanel'
import ClipGallery from './components/ClipGallery'
import VideoPlayer from './components/VideoPlayer'

function App() {
  const [activeFeature, setActiveFeature] = useState('search'); // 'search', 'face', 'prompt'
  const [videoFile, setVideoFile] = useState(null);
  const [clips, setClips] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleFeatureChange = (featureId) => {
    setActiveFeature(featureId);
  };

  const handleVideoUpload = (file) => {
    setVideoFile(file);
    // Don't send right away; we'll send it with the query when processing!
  };

  const handleProcess = async (queryData) => {
    if (!videoFile) {
      alert("Please select a video file first.");
      return;
    }

    setIsLoading(true);
    setClips([]); // Clear previous clips

    let endpoint = '';
    const formData = new FormData();
    formData.append('video', videoFile);

    if (queryData.feature === 'search') {
      endpoint = '/api/dialogue-search';
      formData.append('query', queryData.query);
    } else if (queryData.feature === 'face') {
      endpoint = '/api/face-detection';
      formData.append('reference_image', queryData.referenceImage);
      // In a real scenario you could append 'reference_image'.
      // For now, if no reference image is provided, the backend detects all faces.
    } else if (queryData.feature === 'prompt') {
      endpoint = '/api/prompt-search';
      formData.append('prompt', queryData.query);
    }

    try {
      // Because we have the proxy in vite.config.js, we hit /api natively.
      const response = await fetch(endpoint, {
        method: 'POST',
        // Note: Do NOT set Content-Type header when using FormData, 
        // the browser automatically sets it with the boundary!
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }

      const results = await response.json();

      // Update the Gallery with the backend results!
      setClips(results.clips || []);

    } catch (error) {
      console.error("Error processing video:", error);
      alert("Failed to process: " + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeFeature={activeFeature} onFeatureChange={handleFeatureChange} />

      <div className="main-content">
        <div className="dashboard-grid animate-fade-in">

          <div className="content-left">
            <VideoPlayer
              videoFile={videoFile}
              onUpload={handleVideoUpload}
            />
            {isLoading && (
              <div style={{ marginTop: '20px', padding: '16px', color: '#8b5cf6', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '10px' }}>
                Loading and analyzing video... This strictly depends on video size.
              </div>
            )}
            {!isLoading && videoFile && (
              <ClipGallery clips={clips} />
            )}
          </div>

          <div className="content-right">
            <FeaturePanel
              activeFeature={activeFeature}
              onProcess={handleProcess}
              hasVideo={!!videoFile}
            />
          </div>

        </div>
      </div>
    </div>
  )
}

export default App
