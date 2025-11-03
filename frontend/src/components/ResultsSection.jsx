import './ResultsSection.css';

const behaviorEmojis = {
  'Looking_Forward': '👀',
  'Raising_Hand': '✋',
  'Reading': '📖',
  'Sleeping': '😴',
  'Standing': '🧍',
  'Turning_Around': '🔄',
  'Writting': '✍️'
};

const behaviorColors = {
  'Looking_Forward': '#06b6d4',
  'Raising_Hand': '#10b981',
  'Reading': '#3b82f6',
  'Sleeping': '#f59e0b',
  'Standing': '#94a3b8',
  'Turning_Around': '#f43f5e',
  'Writting': '#8b5cf6'
};

function ResultsSection({ results }) {
  if (!results || !results.result) return null;

  const { type, result } = results;

  // Helper guards
  const safeEmoji = (label) => behaviorEmojis[label] || '📌';
  const safeColor = (label) => behaviorColors[label] || '#64748b';

  return (
    <section className="results-section">
      <div className="results-card">
        <h2>📊 Analysis Results</h2>
        
        {type === 'image' && result.top_prediction && (
          <div className="image-results">
            <div className="top-prediction" style={{ borderColor: safeColor(result.top_prediction) }}>
              <div className="prediction-emoji">{safeEmoji(result.top_prediction)}</div>
              <h3>{result.top_prediction}</h3>
              <p className="confidence-text">
                Confidence: <span style={{ color: safeColor(result.top_prediction) }}>
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </p>
            </div>

            {Array.isArray(result.predictions) && (
              <div className="predictions-list">
                <h4>All Predictions</h4>
                {result.predictions.map((pred, idx) => (
                  <div key={idx} className="prediction-item">
                    <div className="prediction-header">
                      <span className="prediction-label">
                        {safeEmoji(pred.label)} {pred.label}
                      </span>
                      <span className="prediction-confidence">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="prediction-bar">
                      <div
                        className="prediction-bar-fill"
                        style={{
                          width: `${pred.confidence * 100}%`,
                          backgroundColor: safeColor(pred.label)
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {type === 'video' && (
          <div className="video-results">
            {/* Old shape support */}
            {typeof result.total_frames === 'number' && (
              <>
                <div className="video-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Frames</span>
                    <span className="stat-value">{result.total_frames}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Analyzed Frames</span>
                    <span className="stat-value">{result.analyzed_frames}</span>
                  </div>
                </div>

                {Array.isArray(result.behavior_stats) && (
                  <>
                    <h3>Behavior Distribution</h3>
                    <div className="behavior-chart">
                      {result.behavior_stats.map((stat, idx) => (
                        <div key={idx} className="behavior-bar">
                          <div className="behavior-bar-header">
                            <span className="behavior-label">
                              {safeEmoji(stat.label)} {stat.label}
                            </span>
                            <span className="behavior-percentage">
                              {stat.percentage.toFixed(1)}%
                            </span>
                          </div>
                          <div className="behavior-bar-container">
                            <div
                              className="behavior-bar-fill"
                              style={{
                                width: `${stat.percentage}%`,
                                backgroundColor: safeColor(stat.label)
                              }}
                            ></div>
                          </div>
                          <span className="behavior-count">{stat.count} frames</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </>
            )}

            {/* New shape from classroom analyzer */}
            {result.student_data && (
              <>
                <div className="video-stats">
                  <div className="stat-item">
                    <span className="stat-label">Detected Students</span>
                    <span className="stat-value">{result.total_students ?? Object.keys(result.student_data).length}</span>
                  </div>
                </div>

                <h3>Per-student Summary</h3>
                <div className="behavior-chart">
                  {Object.entries(result.student_data).map(([studentId, data]) => (
                    <div key={studentId} className="behavior-bar">
                      <div className="behavior-bar-header">
                        <span className="behavior-label">{studentId} - {data.name}</span>
                        <span className="behavior-percentage">{data.frames?.length ?? 0} frames</span>
                      </div>
                      <div className="behavior-bar-container">
                        {/* Show dominant behavior if available */}
                        {Array.isArray(data.behavior_stats) && data.behavior_stats[0] && (
                          <div
                            className="behavior-bar-fill"
                            style={{
                              width: `${data.behavior_stats[0].percentage.toFixed(1)}%`,
                              backgroundColor: safeColor(data.behavior_stats[0].label)
                            }}
                          ></div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

export default ResultsSection;


