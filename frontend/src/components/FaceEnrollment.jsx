import { useState, useRef, useEffect } from 'react';
import './FaceEnrollment.css';
import AdminLogin from './AdminLogin';

function FaceEnrollment() {
  const [studentId, setStudentId] = useState('');
  const [name, setName] = useState('');
  const [capturedFaces, setCapturedFaces] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [enrolledStudents, setEnrolledStudents] = useState([]);
  const [isAdmin, setIsAdmin] = useState(false);
  const [showLogin, setShowLogin] = useState(true);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [bulkFile, setBulkFile] = useState(null);
  const [activeMode, setActiveMode] = useState('webcam'); // webcam, upload, bulk
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    checkAdminStatus();
    loadEnrolledStudents();
  }, []);

  const checkAdminStatus = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/admin/status', {
        credentials: 'include'
      });
      const data = await res.json();
      setIsAdmin(data.logged_in);
    } catch (error) {
      console.error('Error checking admin status:', error);
    }
  };

  const loadEnrolledStudents = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/face/enrolled');
      const data = await res.json();
      setEnrolledStudents(data.enrolled_students || []);
    } catch (error) {
      console.error('Error loading enrolled students:', error);
    }
  };

  const startCapture = async () => {
    if (!studentId || !name) {
      alert('Please enter Student ID and Name');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setIsCapturing(true);
      setCapturedFaces([]);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera');
    }
  };

  const stopCapture = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCapturing(false);
  };

  const captureFace = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    setCapturedFaces(prev => [...prev, imageData]);
  };

  const removeFace = (index) => {
    setCapturedFaces(prev => prev.filter((_, i) => i !== index));
  };

  const enrollStudent = async () => {
    if (capturedFaces.length < 5) {
      alert('Please capture at least 5 face samples');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/face/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          student_id: studentId,
          name: name,
          face_images: capturedFaces
        })
      });

      const result = await response.json();
      
      if (result.success) {
        alert(`Successfully enrolled ${name}!`);
        setStudentId('');
        setName('');
        setCapturedFaces([]);
        stopCapture();
        loadEnrolledStudents();
      } else {
        alert('Enrollment failed: ' + result.error);
      }
    } catch (error) {
      console.error('Error enrolling student:', error);
      alert('Failed to enroll student');
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    setUploadFiles(files);
  };

  const uploadExistingFaces = async () => {
    if (!studentId || !name) {
      alert('Please enter Student ID and Name');
      return;
    }

    if (uploadFiles.length < 5) {
      alert('Please select at least 5 face images');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('student_id', studentId);
      formData.append('name', name);
      
      uploadFiles.forEach((file, idx) => {
        formData.append(`face_${idx}`, file);
      });

      const response = await fetch('http://localhost:5000/api/admin/upload-student-faces', {
        method: 'POST',
        credentials: 'include',
        body: formData
      });

      const result = await response.json();

      if (result.success) {
        alert(result.message);
        setStudentId('');
        setName('');
        setUploadFiles([]);
        loadEnrolledStudents();
      } else {
        alert('Upload failed: ' + result.error);
      }
    } catch (error) {
      console.error('Error uploading faces:', error);
      alert('Failed to upload faces');
    }
  };

  const handleBulkUpload = async () => {
    if (!bulkFile) {
      alert('Please select a ZIP file');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', bulkFile);

      const response = await fetch('http://localhost:5000/api/admin/bulk-upload-faces', {
        method: 'POST',
        credentials: 'include',
        body: formData
      });

      const result = await response.json();

      if (result.success) {
        alert(`${result.message}\nEnrolled: ${result.enrolled_count} students`);
        setBulkFile(null);
        loadEnrolledStudents();
      } else {
        alert('Bulk upload failed: ' + result.error);
      }
    } catch (error) {
      console.error('Error bulk uploading:', error);
      alert('Failed to bulk upload');
    }
  };

  const deleteStudentFaces = async (studentId) => {
    if (!confirm(`Delete face data for ${studentId}?`)) return;

    try {
      const response = await fetch(`http://localhost:5000/api/admin/delete-student-faces/${studentId}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      const result = await response.json();

      if (result.success) {
        alert(result.message);
        loadEnrolledStudents();
      } else {
        alert('Delete failed: ' + result.error);
      }
    } catch (error) {
      console.error('Error deleting faces:', error);
      alert('Failed to delete faces');
    }
  };

  if (!isAdmin) {
    if (showLogin) {
      return (
        <AdminLogin 
          onLoginSuccess={() => {
            setIsAdmin(true);
            setShowLogin(false);
          }} 
          onClose={() => setShowLogin(false)}
        />
      );
    }
    
    return (
      <div className="face-enrollment">
        <div className="enrollment-card">
          <h2>🔒 Admin Access Required</h2>
          <p className="description">
            Face enrollment is restricted to administrators. Please log in to continue.
          </p>
          <button 
            className="btn-primary" 
            onClick={() => setShowLogin(true)}
            style={{ marginTop: '2rem' }}
          >
            🔑 Admin Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="face-enrollment">
      <div className="enrollment-card">
        <h2>👤 Face Enrollment (Admin)</h2>
        <p className="description">
          Enroll students' faces for automatic identification in classroom analysis
        </p>

        {/* Tab Selector */}
        <div className="enrollment-tabs">
          <button 
            className={`tab-btn ${activeMode === 'webcam' ? 'active' : ''}`} 
            onClick={() => setActiveMode('webcam')}
          >
            📸 Webcam Capture
          </button>
          <button 
            className={`tab-btn ${activeMode === 'upload' ? 'active' : ''}`} 
            onClick={() => setActiveMode('upload')}
          >
            📁 Upload Images
          </button>
          <button 
            className={`tab-btn ${activeMode === 'bulk' ? 'active' : ''}`} 
            onClick={() => setActiveMode('bulk')}
          >
            📦 Bulk Upload
          </button>
        </div>

        {/* Webcam Mode */}
        {activeMode === 'webcam' && !isCapturing && (
          <div className="enrollment-form">
            <input
              type="text"
              className="input-field"
              placeholder="Student ID (e.g., S001)"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
            />
            <input
              type="text"
              className="input-field"
              placeholder="Student Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <button className="btn-primary" onClick={startCapture}>
              📸 Start Webcam Capture
            </button>
          </div>
        )}

        {/* Upload Images Mode */}
        {activeMode === 'upload' && (
          <div className="upload-section">
            <h3>📁 Upload Existing Face Images</h3>
            <input
              type="text"
              className="input-field"
              placeholder="Student ID (e.g., S001)"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
            />
            <input
              type="text"
              className="input-field"
              placeholder="Student Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              className="file-input"
              id="file-upload-input"
            />
            <label htmlFor="file-upload-input" className="file-label">
              {uploadFiles.length > 0 
                ? `${uploadFiles.length} images selected` 
                : 'Choose face images (5+ recommended)'}
            </label>
            {uploadFiles.length > 0 && (
              <div className="upload-preview">
                <div className="preview-images">
                  {uploadFiles.slice(0, 5).map((file, idx) => (
                    <div key={idx} className="preview-thumb">
                      <img src={URL.createObjectURL(file)} alt={`Preview ${idx + 1}`} />
                    </div>
                  ))}
                  {uploadFiles.length > 5 && (
                    <div className="preview-more">+{uploadFiles.length - 5} more</div>
                  )}
                </div>
                <button className="btn-upload" onClick={uploadExistingFaces}>
                  Upload {uploadFiles.length} Images
                </button>
              </div>
            )}
          </div>
        )}

        {/* Bulk Upload Mode */}
        {activeMode === 'bulk' && (
          <div className="bulk-upload-section">
            <h3>📦 Bulk Upload (ZIP File)</h3>
            <p className="bulk-help">
              Upload a ZIP file containing folders for each student.<br/>
              Example structure: <code>S001_JohnDoe/face1.jpg, face2.jpg...</code>
            </p>
            <input
              type="file"
              accept=".zip"
              onChange={(e) => setBulkFile(e.target.files[0])}
              className="file-input"
              id="bulk-upload-input"
            />
            <label htmlFor="bulk-upload-input" className="file-label bulk">
              {bulkFile ? `📦 ${bulkFile.name}` : 'Choose ZIP file'}
            </label>
            {bulkFile && (
              <div className="bulk-info">
                <p>✓ File selected: <strong>{bulkFile.name}</strong></p>
                <p>Size: {(bulkFile.size / 1024 / 1024).toFixed(2)} MB</p>
                <button className="btn-bulk" onClick={handleBulkUpload}>
                  📦 Process Bulk Upload
                </button>
              </div>
            )}
          </div>
        )}

        {/* Webcam Capture in Progress */}
        {activeMode === 'webcam' && isCapturing && (
          <>
            <div className="capture-section">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="video-feed"
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>

            <div className="capture-controls">
              <button className="btn-capture" onClick={captureFace}>
                📸 Capture Face ({capturedFaces.length}/10)
              </button>
              <button className="btn-secondary" onClick={stopCapture}>
                Cancel
              </button>
            </div>

            {capturedFaces.length > 0 && (
              <div className="captured-faces">
                <h3>Captured Samples: {capturedFaces.length}</h3>
                <div className="faces-grid">
                  {capturedFaces.map((face, idx) => (
                    <div key={idx} className="face-item">
                      <img src={face} alt={`Face ${idx + 1}`} />
                      <button
                        className="btn-remove-face"
                        onClick={() => removeFace(idx)}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
                
                {capturedFaces.length >= 5 && (
                  <button className="btn-enroll" onClick={enrollStudent}>
                    Enroll {name} with {capturedFaces.length} samples
                  </button>
                )}
              </div>
            )}
          </>
        )}

        <div className="enrolled-section">
          <h3>Enrolled Students ({enrolledStudents.length})</h3>
          {enrolledStudents.length === 0 ? (
            <p className="no-students">No students enrolled yet</p>
          ) : (
            <div className="enrolled-list">
              {enrolledStudents.map((student, idx) => (
                <div key={idx} className="enrolled-item">
                  <span className="enrolled-icon">✓</span>
                  <div className="enrolled-info">
                    <strong>{student.name}</strong>
                    <span>{student.student_id}</span>
                  </div>
                  <button
                    className="btn-delete-small"
                    onClick={() => deleteStudentFaces(student.student_id)}
                    title="Delete face data"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default FaceEnrollment;

