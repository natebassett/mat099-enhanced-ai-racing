CREATE TABLE agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    version TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    code_commit TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (name, version, config_json)
);

CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT
);

CREATE TABLE race_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    agent_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    track TEXT NOT NULL,
    seed INTEGER,
    total_score REAL NOT NULL,
    steps INTEGER NOT NULL,
    max_speed REAL NOT NULL,
    avg_speed REAL NOT NULL,
    off_track_count INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL,
    termination_reason TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE run_metrics (
    run_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    PRIMARY KEY (run_id, metric_name),
    FOREIGN KEY (run_id) REFERENCES race_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_race_runs_agent_score
    ON race_runs(agent_id, total_score DESC);

CREATE INDEX idx_race_runs_experiment
    ON race_runs(experiment_id);

CREATE INDEX idx_race_runs_started_at
    ON race_runs(started_at DESC);

CREATE INDEX idx_race_runs_termination
    ON race_runs(termination_reason);
