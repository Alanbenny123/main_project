import { useState, useRef, useEffect } from 'react';
import './ClassroomAnalysis.css';

function ClassroomAnalysis() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [detections, setDetections] = useState([]);
  const [sessionData, setSessionData] = useState({});
  const [faceStatus, setFaceStatus] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    // Check face recognition status
    fetch('http://localhost:5000/api/face/status')
      .then(res => res.json())
      .then(data => setFaceStatus(data))
      .catch(err => console.error('Error loading face status:', err));
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 }
      });
      
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setIsStreaming(true);
      
      // Initialize session data for each student
      setSessionData({});

      // Start analyzing frames every 3 seconds
      intervalRef.current = setInterval(() => {
        captureAndAnalyze();
      }, 3000);

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
    
    // Save reports for all detected students
    saveAllReports();
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
      const response = await fetch('http://localhost:5000/api/analyze-classroom-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameData })
      });
      
      const result = await response.json();
      
      if (result.success) {
        setDetections(result.detections || []);
        
        // Update session data for each detected student
        setSessionData(prev => {
          const updated = { ...prev };
          
          result.detections.forEach(det => {
            const studentId = det.student_id;
            
            if (!updated[studentId]) {
              updated[studentId] = {
                name: det.name,
                startTime: Date.now(),
                frames: [],
                behaviors: {
                  'Raising Hand': 0,
                  'Reading': 0,
                  'Sleeping': 0,
                  'Writing': 0
                }
              };
            }
            
            updated[studentId].frames.push({
              timestamp: (Date.now() - updated[studentId].startTime) / 1000,
              behavior: det.behavior,
              confidence: det.behavior_confidence
            });
            
            updated[studentId].behaviors[det.behavior]++;
          });
          
          return updated;
        });
      }
    } catch (error) {
      console.error('Error analyzing frame:', error);
    }
  };

  const saveAllReports = async () => {
    for (const [studentId, data] of Object.entries(sessionData)) {
      if (studentId === 'Unknown') continue;
      
      const duration = data.frames.length * 3; // 3 seconds per frame
      const total = Object.values(data.behaviors).reduce((a, b) => a + b, 0);
      
      if (total === 0) continue;
      
      const behaviorStats = Object.entries(data.behaviors).map(([label, count]) => ({
        label,
        count,
        percentage: (count / total) * 100
      }));

      try {
        await fetch('http://localhost:5000/api/reports/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            student_id: studentId,
            duration,
            behavior_stats: behaviorStats,
            frame_data: data.frames,
            notes: 'Classroom analysis session'
          })
        });
      } catch (error) {
        console.error(`Error saving report for ${studentId}:`, error);
      }
    }
    
    alert(`Saved reports for ${Object.keys(sessionData).filter(k => k !== 'Unknown').length} students!`);
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

  return (
    <div className="classroom-analysis">
      <div className="classroom-card">
        <h2>🎥 Full Classroom Analysis</h2>
        <p className="description">Detects and tracks multiple students simultaneously using face recognition</p>
        
        {faceStatus && (
          <div className={`face-status ${faceStatus.available && faceStatus.trained ? 'ready' : 'not-ready'}`}>
            <div className="status-indicator"></div>
            <div className="status-text">
              {faceStatus.available && faceStatus.trained ? (
                <>
                  <strong>Face Recognition Ready</strong>
                  <span>{faceStatus.enrolled_count} students enrolled</span>
                </>
              ) : (
                <>
                  <strong>Face Recognition Not Trained</strong>
                  <span>Please enroll students first</span>
                </>
              )}
            </div>
          </div>
        )}

        <div className="video-section">
          {!isStreaming && (
            <div className="video-placeholder">
              <div className="placeholder-icon">🎥</div>
              <p>Classroom camera feed will appear here</p>
              <p className="placeholder-hint">Click "Start Classroom Analysis" to begin</p>
            </div>
          )}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="classroom-video"
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          
          {isStreaming && detections.length > 0 && (
            <div className="detections-overlay">
              {detections.map((det, idx) => (
                <div key={idx} className="detection-card">
                  <div className="detection-header">
                    <span className="student-name">{det.name}</span>
                    <span className="student-id">{det.student_id}</span>
                  </div>
                  <div className="detection-behavior">
                    {det.behavior}
                  </div>
                  <div className="detection-confidence">
                    {(det.behavior_confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {!isStreaming ? (
          <button className="btn-primary" onClick={startCamera}>
            Start Classroom Analysis
          </button>
        ) : (
          <button className="btn-stop" onClick={stopCamera}>
            Stop & Save All Reports
          </button>
        )}

        {isStreaming && Object.keys(sessionData).length > 0 && (
          <div className="student-tracking">
            <h3>Active Students: {Object.keys(sessionData).length}</h3>
            <div className="tracking-grid">
              {Object.entries(sessionData).map(([studentId, data]) => (
                <div key={studentId} className="tracking-card">
                  <div className="tracking-header">
                    <strong>{data.name}</strong>
                    <span>{studentId}</span>
                  </div>
                  <div className="tracking-stats">
                    <span>Frames: {data.frames.length}</span>
                  </div>
                  <div className="behavior-mini">
                    {Object.entries(data.behaviors).map(([behavior, count]) => (
                      count > 0 && (
                        <div key={behavior} className="behavior-mini-item">
                          <span>{behavior}:</span>
                          <strong>{count}</strong>
                        </div>
                      )
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ClassroomAnalysis;

