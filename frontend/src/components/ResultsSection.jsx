import './ResultsSection.css';

const behaviorEmojis = {
  'Raising Hand': '✋',
  'Reading': '📖',
  'Sleeping': '😴',
  'Writing': '✍️'
};

const behaviorColors = {
  'Raising Hand': '#10b981',
  'Reading': '#3b82f6',
  'Sleeping': '#f59e0b',
  'Writing': '#8b5cf6'
};

function ResultsSection({ results }) {
  if (!results || !results.result) return null;

  const { type, result } = results;

  return (
    <section className="results-section">
      <div className="results-card">
        <h2>📊 Analysis Results</h2>
        
        {type === 'image' && (
          <div className="image-results">
            <div className="top-prediction" style={{ borderColor: behaviorColors[result.top_prediction] }}>
              <div className="prediction-emoji">{behaviorEmojis[result.top_prediction]}</div>
              <h3>{result.top_prediction}</h3>
              <p className="confidence-text">
                Confidence: <span style={{ color: behaviorColors[result.top_prediction] }}>
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </p>
            </div>

            <div className="predictions-list">
              <h4>All Predictions</h4>
              {result.predictions.map((pred, idx) => (
                <div key={idx} className="prediction-item">
                  <div className="prediction-header">
                    <span className="prediction-label">
                      {behaviorEmojis[pred.label]} {pred.label}
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
                        backgroundColor: behaviorColors[pred.label]
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {type === 'video' && (
          <div className="video-results">
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

            <h3>Behavior Distribution</h3>
            <div className="behavior-chart">
              {result.behavior_stats.map((stat, idx) => (
                <div key={idx} className="behavior-bar">
                  <div className="behavior-bar-header">
                    <span className="behavior-label">
                      {behaviorEmojis[stat.label]} {stat.label}
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
                        backgroundColor: behaviorColors[stat.label]
                      }}
                    ></div>
                  </div>
                  <span className="behavior-count">{stat.count} frames</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default ResultsSection;


