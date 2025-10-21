import { useState, useEffect } from 'react';
import './AuditLog.css';
import AdminLogin from './AdminLogin';

function AuditLog() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    checkAdminStatus();
  }, []);

  const checkAdminStatus = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/admin/status', {
        credentials: 'include'
      });
      const data = await res.json();
      if (data.logged_in) {
        setIsAdmin(true);
        loadAuditLogs();
      }
    } catch (error) {
      console.error('Error checking admin status:', error);
    }
  };

  const loadAuditLogs = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/api/admin/audit-logs', {
        credentials: 'include'
      });
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (error) {
      console.error('Error loading audit logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const getActionColor = (action) => {
    if (action.includes('create') || action.includes('enroll')) return '#10b981';
    if (action.includes('delete')) return '#ef4444';
    if (action.includes('login')) return '#3b82f6';
    if (action.includes('logout')) return '#6b7280';
    return '#f59e0b';
  };

  if (!isAdmin) {
    return <AdminLogin onLoginSuccess={() => setIsAdmin(true)} />;
  }

  return (
    <div className="audit-log">
      <div className="audit-card">
        <div className="audit-header">
          <h2>📋 Audit Log</h2>
          <button className="btn-refresh" onClick={loadAuditLogs}>
            🔄 Refresh
          </button>
        </div>

        {loading ? (
          <div className="loading">Loading logs...</div>
        ) : logs.length === 0 ? (
          <div className="no-logs">No activity logs yet</div>
        ) : (
          <div className="logs-table">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>User</th>
                  <th>Type</th>
                  <th>Action</th>
                  <th>Entity</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td className="time-cell">{formatDate(log.timestamp)}</td>
                    <td className="user-cell">
                      <span className={`user-badge ${log.user_type}`}>
                        {log.user_type === 'admin' ? '👑' : '🎓'} {log.user_id}
                      </span>
                    </td>
                    <td>{log.user_type}</td>
                    <td>
                      <span 
                        className="action-badge"
                        style={{ backgroundColor: getActionColor(log.action) }}
                      >
                        {log.action}
                      </span>
                    </td>
                    <td>
                      {log.entity_type && (
                        <span className="entity-info">
                          {log.entity_type}: {log.entity_id}
                        </span>
                      )}
                    </td>
                    <td className="details-cell">{log.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default AuditLog;


