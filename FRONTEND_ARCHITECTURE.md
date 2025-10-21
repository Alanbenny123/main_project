# Frontend Architecture Documentation

Complete guide to the React-based frontend of the Student Behavior Analysis System.

---

## 🏗️ Architecture Overview

**Framework**: React 18 with Vite  
**Language**: JavaScript (ES6+)  
**Styling**: Pure CSS (no frameworks)  
**State Management**: React Hooks (useState, useEffect, useRef)  
**API Communication**: Fetch API with async/await  

---

## 📁 Project Structure

```
frontend/
├── public/
│   └── vite.svg              # App icon
├── src/
│   ├── App.jsx               # Main component (routing & layout)
│   ├── App.css               # Global styles & themes
│   ├── index.css             # Base styles & fonts
│   ├── main.jsx              # React entry point
│   └── components/
│       ├── UploadSection.jsx     # File upload & image/video analysis
│       ├── UploadSection.css
│       ├── ResultsSection.jsx    # Display analysis results
│       ├── ResultsSection.css
│       ├── StatsSection.jsx      # Dataset statistics
│       ├── StatsSection.css
│       ├── LiveAnalysis.jsx      # Single student webcam analysis
│       ├── LiveAnalysis.css
│       ├── ClassroomAnalysis.jsx # Multi-student video analysis
│       ├── ClassroomAnalysis.css
│       ├── FaceEnrollment.jsx    # Admin face enrollment
│       ├── FaceEnrollment.css
│       ├── StudentDashboard.jsx  # Student report viewing
│       ├── StudentDashboard.css
│       ├── AdminLogin.jsx        # Admin authentication modal
│       ├── AdminLogin.css
│       ├── StudentLogin.jsx      # Student authentication modal
│       └── AuditLog.jsx          # Admin audit log viewer
│           └── AuditLog.css
├── package.json              # Dependencies & scripts
├── vite.config.js            # Vite configuration
└── index.html                # HTML entry point
```

---

## 🎨 Design System

### Color Palette

#### Light Theme
```css
--background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)
--primary: #3b82f6      /* Deep Blue */
--secondary: #14b8a6    /* Teal */
--accent: #f59e0b       /* Orange */
--surface: #ffffff      /* Pure White */
--text: #1f2937         /* Dark Gray */
--text-muted: #6b7280   /* Medium Gray */
```

#### Dark Theme
```css
--background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%)
--primary: #60a5fa      /* Light Blue */
--secondary: #2dd4bf    /* Light Teal */
--surface: rgba(30, 41, 59, 0.95)  /* Dark Slate */
--text: #e2e8f0         /* Light Gray */
--text-muted: #94a3b8   /* Muted Gray */
```

### Typography

```css
Font Family: 'Roboto', 'Montserrat', 'Nunito', sans-serif
Headings: 'Montserrat' (700 weight)
Body: 'Roboto' (400 weight)
Buttons: 'Nunito' (600 weight)
```

### Spacing Scale
```css
0.25rem (4px)   → xs
0.5rem (8px)    → sm
1rem (16px)     → md
1.5rem (24px)   → lg
2rem (32px)     → xl
3rem (48px)     → 2xl
```

---

## 🧩 Component Breakdown

### 1. **App.jsx** - Main Application

**Responsibilities**:
- Tab navigation
- Global state management
- Theme switching (light/dark)
- Model status display

**State**:
```javascript
const [activeTab, setActiveTab] = useState('upload')
const [darkMode, setDarkMode] = useState(false)
```

**Tabs**:
- `upload` → UploadSection
- `live` → LiveAnalysis
- `classroom` → ClassroomAnalysis
- `enroll` → FaceEnrollment
- `dashboard` → StudentDashboard
- `audit` → AuditLog

**Key Features**:
- Persistent dark mode (localStorage)
- Responsive tab navigation
- Model status indicator

---

### 2. **UploadSection.jsx** - File Upload & Analysis

**Purpose**: Upload images/videos for behavior analysis

**State**:
```javascript
const [file, setFile] = useState(null)
const [preview, setPreview] = useState(null)
const [analyzing, setAnalyzing] = useState(false)
const [result, setResult] = useState(null)
const [dragActive, setDragActive] = useState(false)
```

**Features**:
- Drag & drop file upload
- Image/video preview
- Progress indicator
- Result visualization

**API Call**:
```javascript
const formData = new FormData()
formData.append('file', file)

const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData
})

const data = await response.json()
setResult(data.result)
```

**Supported Formats**:
- Images: JPG, PNG, JPEG, GIF
- Videos: MP4, AVI, MOV

---

### 3. **LiveAnalysis.jsx** - Real-Time Webcam Analysis

**Purpose**: Single student live webcam monitoring

**State**:
```javascript
const [studentId, setStudentId] = useState('')
const [isActive, setIsActive] = useState(false)
const [currentBehavior, setCurrentBehavior] = useState('')
const [behaviorCounts, setBehaviorCounts] = useState({})
const [sessionData, setSessionData] = useState([])
```

**Refs**:
```javascript
const videoRef = useRef(null)
const canvasRef = useRef(null)
const streamRef = useRef(null)
const intervalRef = useRef(null)
```

**Webcam Flow**:
```javascript
// 1. Start webcam
const stream = await navigator.mediaDevices.getUserMedia({ video: true })
videoRef.current.srcObject = stream

// 2. Capture frame every 2 seconds
setInterval(async () => {
  const canvas = canvasRef.current
  const video = videoRef.current
  canvas.getContext('2d').drawImage(video, 0, 0)
  
  // 3. Convert to base64
  const frameData = canvas.toDataURL('image/jpeg')
  
  // 4. Send to backend
  const response = await fetch('http://localhost:5000/api/analyze-frame', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: frameData })
  })
  
  const data = await response.json()
  updateBehavior(data.behavior)
}, 2000)
```

**Authentication**:
- Admin login required OR
- Student login with own ID

**Report Saving**:
```javascript
const saveReport = async () => {
  await fetch('http://localhost:5000/api/reports/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      student_id: studentId,
      duration: sessionDuration,
      behavior_stats: calculateStats(),
      frame_data: sessionData
    })
  })
}
```

---

### 4. **ClassroomAnalysis.jsx** - Multi-Student Analysis

**Purpose**: Analyze full classroom videos with face recognition

**State**:
```javascript
const [videoFile, setVideoFile] = useState(null)
const [isAnalyzing, setIsAnalyzing] = useState(false)
const [studentResults, setStudentResults] = useState([])
const [isLive, setIsLive] = useState(false)
```

**Features**:
- Upload classroom video
- Live classroom webcam feed
- Multi-student detection (YOLO)
- Per-student behavior tracking
- Individual report generation

**API Call**:
```javascript
const formData = new FormData()
formData.append('file', videoFile)

const response = await fetch('http://localhost:5000/api/analyze-classroom-video', {
  method: 'POST',
  body: formData
})

const data = await response.json()
// data.student_data = { 'S001': {...}, 'S002': {...} }
```

---

### 5. **FaceEnrollment.jsx** - Face Data Management

**Purpose**: Admin-only face enrollment (3 modes)

**State**:
```javascript
const [isAdmin, setIsAdmin] = useState(false)
const [activeMode, setActiveMode] = useState('webcam')  // webcam, upload, bulk
const [studentId, setStudentId] = useState('')
const [name, setName] = useState('')
const [capturedFaces, setCapturedFaces] = useState([])
const [enrolledStudents, setEnrolledStudents] = useState([])
```

**Enrollment Modes**:

#### Mode 1: Webcam Capture
```javascript
// Capture 5-10 face samples from webcam
const captureFace = () => {
  const canvas = canvasRef.current
  const video = videoRef.current
  canvas.getContext('2d').drawImage(video, 0, 0)
  const imageData = canvas.toDataURL('image/jpeg')
  setCapturedFaces([...capturedFaces, imageData])
}
```

#### Mode 2: Upload Images
```javascript
// Upload existing face images
const handleFileUpload = (e) => {
  const files = Array.from(e.target.files)
  setUploadFiles(files)
}

const uploadExistingFaces = async () => {
  const formData = new FormData()
  formData.append('student_id', studentId)
  formData.append('name', name)
  uploadFiles.forEach((file, idx) => {
    formData.append(`face_${idx}`, file)
  })
  
  await fetch('http://localhost:5000/api/admin/upload-student-faces', {
    method: 'POST',
    credentials: 'include',
    body: formData
  })
}
```

#### Mode 3: Bulk ZIP Upload
```javascript
// Upload ZIP: S001_JohnDoe/face1.jpg, face2.jpg...
const handleBulkUpload = async () => {
  const formData = new FormData()
  formData.append('file', bulkFile)
  
  await fetch('http://localhost:5000/api/admin/bulk-upload-faces', {
    method: 'POST',
    credentials: 'include',
    body: formData
  })
}
```

**Access Control**:
```javascript
useEffect(() => {
  checkAdminStatus()
}, [])

const checkAdminStatus = async () => {
  const res = await fetch('http://localhost:5000/api/admin/status', {
    credentials: 'include'
  })
  const data = await res.json()
  setIsAdmin(data.logged_in)
}
```

---

### 6. **StudentDashboard.jsx** - Student Reports

**Purpose**: Students view their own behavior reports

**State**:
```javascript
const [isLoggedIn, setIsLoggedIn] = useState(false)
const [studentId, setStudentId] = useState('')
const [student, setStudent] = useState(null)
const [reports, setReports] = useState([])
const [selectedReport, setSelectedReport] = useState(null)
```

**Authentication Flow**:
```javascript
const handleLoginSuccess = (studentData) => {
  setIsLoggedIn(true)
  setStudentId(studentData.student_id)
  loadStudentData(studentData.student_id)
}

const loadStudentData = async (id) => {
  // Get student info
  const studentRes = await fetch(`http://localhost:5000/api/students/${id}`)
  const studentData = await studentRes.json()
  setStudent(studentData)
  
  // Get reports
  const reportsRes = await fetch(`http://localhost:5000/api/reports/student/${id}`)
  const reportsData = await reportsRes.json()
  setReports(reportsData.reports)
}
```

**Report Display**:
- Summary cards with engagement badges
- Behavior distribution bars
- Frame-by-frame analysis
- Timeline view

---

### 7. **AdminLogin.jsx** - Admin Authentication

**Purpose**: Admin login modal

**State**:
```javascript
const [username, setUsername] = useState('')
const [password, setPassword] = useState('')
const [error, setError] = useState('')
const [loading, setLoading] = useState(false)
```

**Login Flow**:
```javascript
const handleLogin = async (e) => {
  e.preventDefault()
  
  const response = await fetch('http://localhost:5000/api/admin/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ username, password })
  })
  
  const data = await response.json()
  
  if (data.success) {
    onLoginSuccess()  // Callback to parent
  } else {
    setError(data.error)
  }
}
```

**Features**:
- Session-based auth
- Close button (overlay click)
- Error messages
- Default credentials display

---

### 8. **StudentLogin.jsx** - Student Authentication

**Purpose**: Student login modal

Similar to AdminLogin but uses:
- `student_id` instead of `username`
- `/api/student/login` endpoint
- Returns student data on success

---

### 9. **AuditLog.jsx** - Activity Tracking

**Purpose**: Admin-only audit log viewer

**State**:
```javascript
const [isAdmin, setIsAdmin] = useState(false)
const [logs, setLogs] = useState([])
const [loading, setLoading] = useState(false)
```

**Data Fetching**:
```javascript
const loadAuditLogs = async () => {
  const response = await fetch('http://localhost:5000/api/admin/audit-logs?limit=100', {
    credentials: 'include'
  })
  
  const data = await response.json()
  setLogs(data.logs)
}
```

**Display**:
- Table format
- Columns: User, Type, Action, Entity, Details, Timestamp
- Auto-refresh button
- Filterable/searchable (future)

---

## 🎭 State Management

### Local State (useState)
Each component manages its own UI state:
```javascript
const [file, setFile] = useState(null)
const [loading, setLoading] = useState(false)
const [error, setError] = useState('')
```

### Refs (useRef)
For DOM access and timers:
```javascript
const videoRef = useRef(null)         // <video> element
const intervalRef = useRef(null)      // setInterval ID
const streamRef = useRef(null)        // MediaStream
```

### Side Effects (useEffect)
For lifecycle events:
```javascript
useEffect(() => {
  // On mount
  loadData()
  
  // Cleanup on unmount
  return () => {
    stopWebcam()
  }
}, [])
```

### Persistent State (localStorage)
For theme preference:
```javascript
// Save
localStorage.setItem('darkMode', 'true')

// Load
const savedTheme = localStorage.getItem('darkMode') === 'true'
setDarkMode(savedTheme)
```

---

## 🌐 API Integration

### Base URL
```javascript
const API_URL = 'http://localhost:5000'
```

### Common Patterns

#### GET Request
```javascript
const response = await fetch(`${API_URL}/api/students/${studentId}`)
const data = await response.json()
```

#### POST with JSON
```javascript
const response = await fetch(`${API_URL}/api/reports/save`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ student_id, duration, behavior_stats })
})
```

#### POST with FormData (file upload)
```javascript
const formData = new FormData()
formData.append('file', file)

const response = await fetch(`${API_URL}/api/predict`, {
  method: 'POST',
  body: formData  // No Content-Type header!
})
```

#### Authenticated Requests
```javascript
const response = await fetch(`${API_URL}/api/admin/audit-logs`, {
  credentials: 'include'  // Send session cookie
})
```

### Error Handling
```javascript
try {
  const response = await fetch(url)
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  
  const data = await response.json()
  
  if (data.error) {
    setError(data.error)
  }
} catch (err) {
  setError('Failed to connect to server')
  console.error(err)
}
```

---

## 🎨 Styling Architecture

### CSS Modules Approach
Each component has its own CSS file:
```
Component.jsx  →  Component.css
```

### Global Styles (App.css)
- Background gradients
- Header/footer styles
- Tab navigation
- Theme transitions

### Component Styles (Component.css)
- Scoped to component
- BEM-like naming: `.component-card`, `.component-header`
- Dark mode variants: `body.dark-mode .component-card`

### Transitions
```css
.card {
  transition: all 0.3s ease;
}

body.dark-mode .card {
  background: #1e293b;
}
```

### Responsive Design
```css
@media (max-width: 768px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
```

---

## 🔄 User Flows

### Flow 1: Upload Image Analysis
```
1. User selects Upload tab
2. Drags image file to drop zone
3. Image preview appears
4. Clicks "Analyze"
5. Loading spinner shows
6. Result displays with behavior + confidence
```

### Flow 2: Live Student Analysis
```
1. User selects Live Analysis tab
2. Logs in (admin or student)
3. Enters Student ID
4. Clicks "Start Analysis"
5. Webcam permission requested
6. Video feed starts
7. Behavior updates every 2 seconds
8. Counters increment
9. Clicks "Stop & Save Report"
10. Report saved to database
```

### Flow 3: Face Enrollment (Admin)
```
1. User selects Face Enrollment tab
2. Admin login modal appears
3. Enters admin credentials
4. Chooses enrollment mode (webcam/upload/bulk)
5. Enters Student ID + Name
6. Captures/uploads face images
7. Clicks "Enroll"
8. InsightFace processes embeddings
9. Success message shown
10. Student appears in enrolled list
```

### Flow 4: View Reports (Student)
```
1. User selects Dashboard tab
2. Student login modal appears
3. Enters Student ID + password
4. Dashboard loads with report cards
5. Clicks "View Details" on a report
6. Modal shows full analysis + timeline
7. Can browse other reports
```

---

## ⚡ Performance Optimizations

### 1. Lazy Loading
```javascript
// Future: Code splitting
const ClassroomAnalysis = lazy(() => import('./components/ClassroomAnalysis'))
```

### 2. Debouncing
```javascript
// Prevent rapid API calls
const debounce = (func, delay) => {
  let timer
  return (...args) => {
    clearTimeout(timer)
    timer = setTimeout(() => func(...args), delay)
  }
}
```

### 3. Image Compression
```javascript
// Reduce upload size
const compressImage = (canvas, quality = 0.8) => {
  return canvas.toDataURL('image/jpeg', quality)
}
```

### 4. Cleanup
```javascript
useEffect(() => {
  return () => {
    // Stop webcam
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }
    
    // Clear intervals
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }
}, [])
```

---

## 🐛 Debugging Tools

### React DevTools
```bash
# Install browser extension
# Chrome: https://chrome.google.com/webstore
# Firefox: https://addons.mozilla.org
```

### Console Logging
```javascript
console.log('State:', { studentId, reports })
console.error('API Error:', error)
console.table(behaviorCounts)
```

### Network Tab
- Check API requests/responses
- Verify headers & payloads
- Monitor timing

---

## 🚀 Build & Deploy

### Development
```bash
npm run dev
# Runs on http://localhost:3000
```

### Production Build
```bash
npm run build
# Output: dist/
```

### Preview Production
```bash
npm run preview
```

### Deploy to Netlify/Vercel
```bash
# Update API_URL to production backend
# Deploy dist/ folder
```

---

## 📦 Dependencies

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "vite": "^4.4.5"
}
```

**No external UI libraries!** Pure CSS for maximum control.

---

## 🔐 Security Considerations

### 1. Session Credentials
```javascript
fetch(url, { credentials: 'include' })
```

### 2. Input Validation
```javascript
if (!studentId || studentId.length < 3) {
  setError('Invalid Student ID')
  return
}
```

### 3. XSS Prevention
- React auto-escapes JSX
- Avoid `dangerouslySetInnerHTML`

### 4. CORS
Backend enables CORS:
```python
CORS(app, supports_credentials=True)
```

---

## 📚 Further Reading

- [React Docs](https://react.dev)
- [Vite Guide](https://vitejs.dev/guide)
- [MDN Web APIs](https://developer.mozilla.org/en-US/docs/Web/API)
- [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

---

**Next**: See [BACKEND_ARCHITECTURE.md](BACKEND_ARCHITECTURE.md) for server-side details.

