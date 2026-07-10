CREATE TABLE run_telemetry (
    run_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    dist_from_start REAL,
    dist_raced REAL,
    speed_x REAL,
    speed_y REAL,
    angle REAL,
    track_pos REAL,
    damage REAL,
    steer REAL,
    accel REAL,
    brake REAL,
    gear INTEGER,
    reward REAL,
    off_track INTEGER NOT NULL DEFAULT 0,
    front_sensor REAL,
    min_track_sensor REAL,
    cur_lap_time REAL,
    last_lap_time REAL,
    PRIMARY KEY (run_id, step),
    FOREIGN KEY (run_id) REFERENCES race_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_run_telemetry_run_distance
    ON run_telemetry(run_id, dist_from_start);

CREATE INDEX idx_run_telemetry_run_track_pos
    ON run_telemetry(run_id, track_pos);
