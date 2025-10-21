import { useState, useRef } from 'react';
import './UploadSection.css';

function UploadSection({ onAnalysisComplete }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'video/mp4', 'video/avi', 'video/quicktime'];
    
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid image or video file');
      return;
    }

    setFile(file);
    const type = file.type.startsWith('video') ? 'video' : 'image';
    setFileType(type);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleRemove = () => {
    setFile(null);
    setPreview(null);
    setFileType(null);
    onAnalysisComplete(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.success) {
        onAnalysisComplete(data);
      } else {
        alert('Analysis failed: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to analyze. Make sure the Flask backend is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="upload-section">
      <div className="upload-card">
        <div className="upload-icon">📤</div>
        <h2>Upload Image or Video</h2>
        <p className="upload-description">Supports images (JPG, PNG) and videos (MP4, AVI, MOV)</p>
        
        {!file ? (
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleChange}
              style={{ display: 'none' }}
            />
            <div className="upload-content">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <p><strong>Click to upload</strong> or drag and drop</p>
              <p className="upload-hint">Maximum file size: 16MB</p>
            </div>
          </div>
        ) : (
          <div className="preview-container">
            {fileType === 'image' ? (
              <img src={preview} alt="Preview" className="preview-media" />
            ) : (
              <video src={preview} controls className="preview-media" />
            )}
            <button className="btn-remove" onClick={handleRemove}>
              ✕ Remove
            </button>
          </div>
        )}

        <button
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={!file || isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <span className="btn-loader">⏳</span>
              Analyzing...
            </>
          ) : (
            'Analyze Behavior'
          )}
        </button>
      </div>
    </section>
  );
}

export default UploadSection;


