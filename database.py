import sqlite3
from datetime import datetime
import json

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    # Students table (now with authentication)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Reports table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration INTEGER,
            total_frames INTEGER,
            raising_hand_count INTEGER DEFAULT 0,
            reading_count INTEGER DEFAULT 0,
            sleeping_count INTEGER DEFAULT 0,
            writing_count INTEGER DEFAULT 0,
            raising_hand_percent REAL DEFAULT 0,
            reading_percent REAL DEFAULT 0,
            sleeping_percent REAL DEFAULT 0,
            writing_percent REAL DEFAULT 0,
            engagement_score REAL,
            notes TEXT,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # Frame analysis table (detailed per-frame data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS frame_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL,
            frame_number INTEGER,
            timestamp REAL,
            behavior TEXT,
            confidence REAL,
            FOREIGN KEY (report_id) REFERENCES reports (id)
        )
    ''')
    
    # Audit log table (track who did what)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_type TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[OK] Database initialized successfully!")

def add_student(student_id, name, email=None, password=None):
    """Add a new student"""
    import hashlib
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    try:
        password_hash = None
        if password:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute(
            'INSERT INTO students (student_id, name, email, password_hash) VALUES (?, ?, ?, ?)',
            (student_id, name, email, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Student already exists
    finally:
        conn.close()

def verify_student_password(student_id, password):
    """Verify student password"""
    import hashlib
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT password_hash FROM students WHERE student_id = ?', (student_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result or not result[0]:
        return False
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == result[0]

def add_audit_log(user_id, user_type, action, entity_type=None, entity_id=None, details=None):
    """Add audit log entry"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO audit_log (user_id, user_type, action, entity_type, entity_id, details)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, user_type, action, entity_type, entity_id, details))
    
    conn.commit()
    conn.close()

def get_audit_logs(limit=100):
    """Get recent audit logs"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM audit_log 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    logs = cursor.fetchall()
    conn.close()
    
    return [
        {
            'id': log[0],
            'user_id': log[1],
            'user_type': log[2],
            'action': log[3],
            'entity_type': log[4],
            'entity_id': log[5],
            'details': log[6],
            'timestamp': log[7]
        }
        for log in logs
    ]

def get_student(student_id):
    """Get student information"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
    student = cursor.fetchone()
    conn.close()
    
    if student:
        return {
            'id': student[0],
            'student_id': student[1],
            'name': student[2],
            'email': student[3],
            'created_at': student[4]
        }
    return None

def save_report(student_id, duration, behavior_stats, frame_data, notes=None):
    """Save an analysis report"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    total_frames = sum(stat['count'] for stat in behavior_stats)
    
    # Calculate behavior counts and percentages
    behavior_counts = {stat['label']: stat['count'] for stat in behavior_stats}
    behavior_percents = {stat['label']: stat['percentage'] for stat in behavior_stats}
    
    # Calculate engagement score (higher for active behaviors)
    engagement_score = (
        behavior_percents.get('Raising Hand', 0) * 1.0 +
        behavior_percents.get('Writing', 0) * 0.9 +
        behavior_percents.get('Reading', 0) * 0.8 +
        behavior_percents.get('Sleeping', 0) * 0.1
    ) / 100
    
    # Insert report
    cursor.execute('''
        INSERT INTO reports (
            student_id, duration, total_frames,
            raising_hand_count, reading_count, sleeping_count, writing_count,
            raising_hand_percent, reading_percent, sleeping_percent, writing_percent,
            engagement_score, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        student_id, duration, total_frames,
        behavior_counts.get('Raising Hand', 0),
        behavior_counts.get('Reading', 0),
        behavior_counts.get('Sleeping', 0),
        behavior_counts.get('Writing', 0),
        behavior_percents.get('Raising Hand', 0),
        behavior_percents.get('Reading', 0),
        behavior_percents.get('Sleeping', 0),
        behavior_percents.get('Writing', 0),
        engagement_score,
        notes
    ))
    
    report_id = cursor.lastrowid
    
    # Insert frame analysis data
    for frame in frame_data:
        cursor.execute('''
            INSERT INTO frame_analysis (report_id, frame_number, timestamp, behavior, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            report_id,
            frame.get('frame', 0),
            frame.get('timestamp', 0),
            frame.get('prediction', ''),
            frame.get('confidence', 0)
        ))
    
    conn.commit()
    conn.close()
    
    return report_id

def get_student_reports(student_id):
    """Get all reports for a student"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM reports 
        WHERE student_id = ? 
        ORDER BY session_date DESC
    ''', (student_id,))
    
    reports = cursor.fetchall()
    conn.close()
    
    return [
        {
            'id': r[0],
            'student_id': r[1],
            'session_date': r[2],
            'duration': r[3],
            'total_frames': r[4],
            'behaviors': {
                'raising_hand': {'count': r[5], 'percent': r[9]},
                'reading': {'count': r[6], 'percent': r[10]},
                'sleeping': {'count': r[7], 'percent': r[11]},
                'writing': {'count': r[8], 'percent': r[12]}
            },
            'engagement_score': r[13],
            'notes': r[14]
        }
        for r in reports
    ]

def get_report_details(report_id):
    """Get detailed report with frame analysis"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    # Get report
    cursor.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    report = cursor.fetchone()
    
    if not report:
        conn.close()
        return None
    
    # Get frame analysis
    cursor.execute('''
        SELECT frame_number, timestamp, behavior, confidence 
        FROM frame_analysis 
        WHERE report_id = ?
        ORDER BY frame_number
    ''', (report_id,))
    
    frames = cursor.fetchall()
    conn.close()
    
    return {
        'id': report[0],
        'student_id': report[1],
        'session_date': report[2],
        'duration': report[3],
        'total_frames': report[4],
        'behaviors': {
            'raising_hand': {'count': report[5], 'percent': report[9]},
            'reading': {'count': report[6], 'percent': report[10]},
            'sleeping': {'count': report[7], 'percent': report[11]},
            'writing': {'count': report[8], 'percent': report[12]}
        },
        'engagement_score': report[13],
        'notes': report[14],
        'frame_analysis': [
            {
                'frame': f[0],
                'timestamp': f[1],
                'behavior': f[2],
                'confidence': f[3]
            }
            for f in frames
        ]
    }

def get_all_students():
    """Get all students"""
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students ORDER BY name')
    students = cursor.fetchall()
    conn.close()
    
    return [
        {
            'id': s[0],
            'student_id': s[1],
            'name': s[2],
            'email': s[3],
            'created_at': s[4]
        }
        for s in students
    ]

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Add sample students with passwords
    add_student('S001', 'John Doe', 'john@example.com', 'password123')
    add_student('S002', 'Jane Smith', 'jane@example.com', 'password123')
    add_student('S003', 'Bob Johnson', 'bob@example.com', 'password123')
    
    print("[OK] Sample students added! (Default password: password123)")

