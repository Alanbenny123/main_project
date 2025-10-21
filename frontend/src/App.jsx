import { useState, useEffect } from 'react';
import './App.css';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import StatsSection from './components/StatsSection';
import LiveAnalysis from './components/LiveAnalysis';
import StudentDashboard from './components/StudentDashboard';
import ClassroomAnalysis from './components/ClassroomAnalysis';
import FaceEnrollment from './components/FaceEnrollment';
import AuditLog from './components/AuditLog';

function App() {
  const [modelStatus, setModelStatus] = useState({ loaded: false, text: 'Loading...' });
  const [results, setResults] = useState(null);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('upload'); // upload, live, classroom, enroll, dashboard, audit
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    // Check model status on load
    fetch('http://localhost:5000/api/model-info')
      .then(res => res.json())
      .then(data => {
        setModelStatus({
          loaded: data.model_loaded,
          text: data.model_loaded ? 'Model Ready' : 'Model Not Loaded (Demo Mode)'
        });
      })
      .catch(() => {
        setModelStatus({ loaded: false, text: 'API Offline' });
      });

    // Load stats
    fetch('http://localhost:5000/api/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error('Failed to load stats:', err));
  }, []);

  useEffect(() => {
    // Apply dark mode class to body
    if (darkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
    // Save preference
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleAnalysisComplete = (data) => {
    setResults(data);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">🎓 Student Behavior Analysis</h1>
          <p className="subtitle">AI-powered classroom behavior detection using deep learning</p>
        </div>
        <div className="header-actions">
          <div className={`model-status ${modelStatus.loaded ? 'active' : ''}`}>
            <span className="status-indicator"></span>
            <span className="status-text">{modelStatus.text}</span>
          </div>
          <button className="theme-toggle" onClick={toggleDarkMode} title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
            {darkMode ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📤 Upload
        </button>
        <button
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          📹 Single Student
        </button>
        <button
          className={`tab-btn ${activeTab === 'classroom' ? 'active' : ''}`}
          onClick={() => setActiveTab('classroom')}
        >
          🎥 Full Classroom
        </button>
        <button
          className={`tab-btn ${activeTab === 'enroll' ? 'active' : ''}`}
          onClick={() => setActiveTab('enroll')}
        >
          👤 Face Enrollment
        </button>
        <button
          className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Reports
        </button>
        <button
          className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => setActiveTab('audit')}
        >
          📋 Audit Log
        </button>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {activeTab === 'upload' && (
          <>
            <UploadSection onAnalysisComplete={handleAnalysisComplete} />
            {results && <ResultsSection results={results} />}
            {stats && <StatsSection stats={stats} />}
          </>
        )}

        {activeTab === 'live' && <LiveAnalysis />}

        {activeTab === 'classroom' && <ClassroomAnalysis />}

        {activeTab === 'enroll' && <FaceEnrollment />}

        {activeTab === 'dashboard' && <StudentDashboard />}

        {activeTab === 'audit' && <AuditLog />}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by Swin Transformer • TensorFlow • Computer Vision</p>
      </footer>
    </div>
  );
}

export default App;
