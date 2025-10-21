import './StatsSection.css';

function StatsSection({ stats }) {
  return (
    <section className="stats-section">
      <h2>📈 Dataset Overview</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-info">
            <h3>{stats.total_samples}</h3>
            <p>Total Samples</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🎯</div>
          <div className="stat-info">
            <h3>{stats.train_samples}</h3>
            <p>Training Samples</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">✅</div>
          <div className="stat-info">
            <h3>{stats.validation_samples}</h3>
            <p>Validation Samples</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🧪</div>
          <div className="stat-info">
            <h3>{stats.test_samples}</h3>
            <p>Test Samples</p>
          </div>
        </div>
      </div>

      <div className="classes-info">
        <h3>Behavior Classes</h3>
        <div className="classes-grid">
          <div className="class-item raising-hand">
            <span className="class-emoji">✋</span>
            <span className="class-name">Raising Hand</span>
          </div>
          <div className="class-item reading">
            <span className="class-emoji">📖</span>
            <span className="class-name">Reading</span>
          </div>
          <div className="class-item sleeping">
            <span className="class-emoji">😴</span>
            <span className="class-name">Sleeping</span>
          </div>
          <div className="class-item writing">
            <span className="class-emoji">✍️</span>
            <span className="class-name">Writing</span>
          </div>
        </div>
      </div>
    </section>
  );
}

export default StatsSection;


