import { useState, useRef, useEffect } from 'react';
import './LiveAnalysis.css';

function LiveAnalysis() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [studentId, setStudentId] = useState('');
  const [sessionData, setSessionData] = useState({
    startTime: null,
    frameData: [],
    behaviorCounts: {
      'Raising Hand': 0,
      'Reading': 0,
      'Sleeping': 0,
      'Writing': 0
    }
  });
  const [currentBehavior, setCurrentBehavior] = useState(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [notes, setNotes] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const startCamera = async () => {
    if (!studentId.trim()) {
      alert('Please enter Student ID first');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setIsStreaming(true);
      
      // Initialize session
      setSessionData({
        startTime: Date.now(),
        frameData: [],
        behaviorCounts: {
          'Raising Hand': 0,
          'Reading': 0,
          'Sleeping': 0,
          'Writing': 0
        }
      });

      // Start analyzing frames every 2 seconds
      intervalRef.current = setInterval(() => {
        captureAndAnalyze();
      }, 2000);

    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setIsStreaming(false);
    setShowSaveModal(true);
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
      const response = await fetch('http://localhost:5000/api/analyze-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameData })
      });
      
      const result = await response.json();
      
      if (result.success) {
        const behavior = result.behavior;
        const confidence = result.confidence;
        
        setCurrentBehavior({ behavior, confidence });
        
        // Update session data
        setSessionData(prev => ({
          ...prev,
          frameData: [...prev.frameData, {
            frame: prev.frameData.length,
            timestamp: (Date.now() - prev.startTime) / 1000,
            prediction: behavior,
            confidence: confidence
          }],
          behaviorCounts: {
            ...prev.behaviorCounts,
            [behavior]: prev.behaviorCounts[behavior] + 1
          }
        }));
      }
    } catch (error) {
      console.error('Error analyzing frame:', error);
    }
  };

  const saveReport = async () => {
    const duration = Math.floor((Date.now() - sessionData.startTime) / 1000);
    const totalFrames = sessionData.frameData.length;
    
    const behaviorStats = Object.entries(sessionData.behaviorCounts).map(([label, count]) => ({
      label,
      count,
      percentage: (count / totalFrames) * 100
    }));

    try {
      const response = await fetch('http://localhost:5000/api/reports/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id: studentId,
          duration,
          behavior_stats: behaviorStats,
          frame_data: sessionData.frameData,
          notes
        })
      });

      const result = await response.json();
      
      if (result.success) {
        alert(`Report saved successfully! Report ID: ${result.report_id}`);
        setShowSaveModal(false);
        // Reset
        setStudentId('');
        setNotes('');
        setCurrentBehavior(null);
      } else {
        alert('Error saving report: ' + result.error);
      }
    } catch (error) {
      console.error('Error saving report:', error);
      alert('Failed to save report');
    }
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const getTotalFrames = () => sessionData.frameData.length;
  const getDuration = () => sessionData.startTime 
    ? Math.floor((Date.now() - sessionData.startTime) / 1000)
    : 0;

  return (
    <div className="live-analysis">
      <div className="live-card">
        <h2>📹 Live Behavior Analysis</h2>
        
        {!isStreaming && (
          <div className="setup-section">
            <input
              type="text"
              className="student-input"
              placeholder="Enter Student ID (e.g., S001)"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
            />
            <button className="btn-primary" onClick={startCamera}>
              Start Camera
            </button>
          </div>
        )}

        <div className="video-section">
          {!isStreaming && (
            <div className="video-placeholder">
              <div className="placeholder-icon">📹</div>
              <p>Camera feed will appear here</p>
              <p className="placeholder-hint">Click "Start Camera" to begin</p>
            </div>
          )}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="video-feed"
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          
          {isStreaming && currentBehavior && (
            <div className="live-prediction">
              <div className="prediction-badge">
                {currentBehavior.behavior}
              </div>
              <div className="prediction-confidence">
                {(currentBehavior.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
          )}
        </div>

        {isStreaming && (
          <>
            <div className="session-info">
              <div className="info-item">
                <span>Duration:</span>
                <strong>{getDuration()}s</strong>
              </div>
              <div className="info-item">
                <span>Frames Analyzed:</span>
                <strong>{getTotalFrames()}</strong>
              </div>
            </div>

            <div className="behavior-counters">
              {Object.entries(sessionData.behaviorCounts).map(([behavior, count]) => (
                <div key={behavior} className="counter-item">
                  <span>{behavior}</span>
                  <span className="count">{count}</span>
                </div>
              ))}
            </div>

            <button className="btn-stop" onClick={stopCamera}>
              Stop & Save Report
            </button>
          </>
        )}
      </div>

      {showSaveModal && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Save Analysis Report</h3>
            <p>Student ID: <strong>{studentId}</strong></p>
            <p>Duration: <strong>{getDuration()}s</strong></p>
            <p>Total Frames: <strong>{getTotalFrames()}</strong></p>
            
            <textarea
              className="notes-input"
              placeholder="Add notes (optional)..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows="4"
            />
            
            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setShowSaveModal(false)}>
                Cancel
              </button>
              <button className="btn-primary" onClick={saveReport}>
                Save Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default LiveAnalysis;

