import { useState, useEffect } from 'react';
import './StudentDashboard.css';
import StudentLogin from './StudentLogin';

function StudentDashboard() {
  const [studentId, setStudentId] = useState('');
  const [student, setStudent] = useState(null);
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loggedInStudent, setLoggedInStudent] = useState(null);
  const [showLogin, setShowLogin] = useState(true);

  useEffect(() => {
    checkLoginStatus();
  }, []);

  const checkLoginStatus = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/student/status', {
        credentials: 'include'
      });
      const data = await res.json();
      if (data.logged_in) {
        setIsLoggedIn(true);
        setStudentId(data.student_id);
        loadStudentData(data.student_id);
      }
    } catch (error) {
      console.error('Error checking login status:', error);
    }
  };

  const handleLoginSuccess = (studentData) => {
    setIsLoggedIn(true);
    setLoggedInStudent(studentData);
    setStudentId(studentData.student_id);
    loadStudentData(studentData.student_id);
  };

  const loadStudentData = async (id = studentId) => {
    if (!id) {
      alert('Please enter Student ID');
      return;
    }

    setLoading(true);
    try {
      // Load student info
      const studentRes = await fetch(`http://localhost:5000/api/students/${id}`);
      if (studentRes.ok) {
        const studentData = await studentRes.json();
        setStudent(studentData);

        // Load reports
        const reportsRes = await fetch(`http://localhost:5000/api/reports/student/${id}`);
        const reportsData = await reportsRes.json();
        setReports(reportsData.reports || []);
      } else {
        alert('Student not found');
        setStudent(null);
        setReports([]);
      }
    } catch (error) {
      console.error('Error loading student data:', error);
      alert('Failed to load student data');
    } finally {
      setLoading(false);
    }
  };

  const viewReportDetails = async (reportId) => {
    try {
      const res = await fetch(`http://localhost:5000/api/reports/${reportId}`);
      const data = await res.json();
      setSelectedReport(data);
    } catch (error) {
      console.error('Error loading report details:', error);
    }
  };

  const getEngagementLevel = (score) => {
    if (score >= 0.8) return { level: 'Excellent', color: '#10b981' };
    if (score >= 0.6) return { level: 'Good', color: '#3b82f6' };
    if (score >= 0.4) return { level: 'Fair', color: '#f59e0b' };
    return { level: 'Needs Improvement', color: '#ef4444' };
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const handleDownloadReport = async (reportId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/reports/${reportId}/download`);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Get filename from Content-Disposition header
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'report.json';
        if (contentDisposition) {
          const matches = /filename="?([^"]+)"?/i.exec(contentDisposition);
          if (matches && matches[1]) {
            filename = matches[1];
          }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        alert('Failed to download report');
      }
    } catch (error) {
      console.error('Error downloading report:', error);
      alert('Failed to download report');
    }
  };

  const handleDownloadAllReports = async () => {
    if (reports.length === 0) {
      alert('No reports to download');
      return;
    }

    try {
      const reportIds = reports.map(r => r.id);
      const response = await fetch('http://localhost:5000/api/reports/download-multiple', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_ids: reportIds })
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'student_reports.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        alert('Failed to download reports');
      }
    } catch (error) {
      console.error('Error downloading reports:', error);
      alert('Failed to download reports');
    }
  };

  if (!isLoggedIn) {
    if (showLogin) {
      return (
        <StudentLogin 
          onLoginSuccess={(studentData) => {
            handleLoginSuccess(studentData);
            setShowLogin(false);
          }}
          onClose={() => setShowLogin(false)}
        />
      );
    }
    
    return (
      <div className="dashboard">
        <div className="dashboard-card">
          <h2>🎓 Student Dashboard</h2>
          <p className="description" style={{ textAlign: 'center', marginBottom: '2rem' }}>
            View your behavior analysis reports and track your progress.
          </p>
          <div style={{ textAlign: 'center' }}>
            <button 
              className="btn-primary" 
              onClick={() => setShowLogin(true)}
            >
              🔑 Student Login
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-card">
        <h2>👤 My Reports</h2>

        {!student && (
          <div className="search-section">
            <button
              className="btn-primary"
              onClick={() => loadStudentData()}
              disabled={loading}
            >
              {loading ? 'Loading...' : 'Load My Reports'}
            </button>
          </div>
        )}

        {student && (
          <div className="student-info">
            <h3>{student.name}</h3>
            <p>Student ID: <strong>{student.student_id}</strong></p>
            {student.email && <p>Email: {student.email}</p>}
            <p className="report-count">Total Reports: <strong>{reports.length}</strong></p>
          </div>
        )}

        {reports.length > 0 && (
          <div className="reports-section">
            <div className="reports-header">
              <h3>Analysis Reports</h3>
              <button 
                className="btn-secondary btn-download-all"
                onClick={handleDownloadAllReports}
                title="Download All Reports"
              >
                ⬇️ Download All
              </button>
            </div>
            <div className="reports-grid">
              {reports.map((report) => {
                const engagement = getEngagementLevel(report.engagement_score);
                return (
                  <div key={report.id} className="report-card">
                    <div className="report-header">
                      <span className="report-date">{formatDate(report.session_date)}</span>
                      <span
                        className="engagement-badge"
                        style={{ backgroundColor: engagement.color }}
                      >
                        {engagement.level}
                      </span>
                    </div>

                    <div className="report-stats">
                      <div className="stat">
                        <span>Duration</span>
                        <strong>{formatDuration(report.duration)}</strong>
                      </div>
                      <div className="stat">
                        <span>Frames</span>
                        <strong>{report.total_frames}</strong>
                      </div>
                    </div>

                    <div className="behavior-summary">
                      {Object.entries(report.behaviors).map(([behavior, data]) => (
                        <div key={behavior} className="behavior-item">
                          <span className="behavior-name">{behavior.replace('_', ' ')}</span>
                          <div className="behavior-bar-mini">
                            <div
                              className="behavior-bar-fill"
                              style={{ width: `${data.percent}%` }}
                            />
                          </div>
                          <span className="behavior-percent">{data.percent.toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>

                    <div className="report-actions">
                      <button
                        className="btn-view-details"
                        onClick={() => viewReportDetails(report.id)}
                      >
                        View Details
                      </button>
                      <button
                        className="btn-download"
                        onClick={() => handleDownloadReport(report.id)}
                        title="Download Report"
                      >
                        ⬇️ Download
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {student && reports.length === 0 && (
          <div className="no-reports">
            <p>No reports found for this student.</p>
          </div>
        )}
      </div>

      {selectedReport && (
        <div className="modal-overlay" onClick={() => setSelectedReport(null)}>
          <div className="modal-large" onClick={(e) => e.stopPropagation()}>
            <button
              className="modal-close"
              onClick={() => setSelectedReport(null)}
            >
              ✕
            </button>

            <h3>Detailed Report</h3>
            <p className="report-date">{formatDate(selectedReport.session_date)}</p>

            <div className="detail-grid">
              <div className="detail-item">
                <span>Duration</span>
                <strong>{formatDuration(selectedReport.duration)}</strong>
              </div>
              <div className="detail-item">
                <span>Total Frames</span>
                <strong>{selectedReport.total_frames}</strong>
              </div>
              <div className="detail-item">
                <span>Engagement Score</span>
                <strong>{(selectedReport.engagement_score * 100).toFixed(1)}%</strong>
              </div>
            </div>

            <h4>Behavior Distribution</h4>
            <div className="behavior-details">
              {Object.entries(selectedReport.behaviors).map(([behavior, data]) => (
                <div key={behavior} className="behavior-detail-item">
                  <div className="behavior-detail-header">
                    <span className="behavior-name">{behavior.replace('_', ' ')}</span>
                    <span className="behavior-count">{data.count} frames ({data.percent.toFixed(1)}%)</span>
                  </div>
                  <div className="behavior-bar-large">
                    <div
                      className="behavior-bar-fill"
                      style={{ width: `${data.percent}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {selectedReport.notes && (
              <div className="notes-section">
                <h4>Notes</h4>
                <p>{selectedReport.notes}</p>
              </div>
            )}

            {selectedReport.frame_analysis && selectedReport.frame_analysis.length > 0 && (
              <div className="frame-analysis-section">
                <h4>Frame-by-Frame Analysis (Sample)</h4>
                <div className="frame-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Frame</th>
                        <th>Time (s)</th>
                        <th>Behavior</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedReport.frame_analysis.slice(0, 10).map((frame, idx) => (
                        <tr key={idx}>
                          <td>{frame.frame}</td>
                          <td>{frame.timestamp.toFixed(1)}</td>
                          <td>{frame.behavior}</td>
                          <td>{(frame.confidence * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {selectedReport.frame_analysis.length > 10 && (
                    <p className="table-note">Showing first 10 of {selectedReport.frame_analysis.length} frames</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default StudentDashboard;

