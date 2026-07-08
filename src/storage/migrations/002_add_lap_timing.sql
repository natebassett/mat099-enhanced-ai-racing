ALTER TABLE race_runs
ADD COLUMN laps_completed INTEGER NOT NULL DEFAULT 0;

ALTER TABLE race_runs
ADD COLUMN best_lap_time_seconds REAL;

ALTER TABLE race_runs
ADD COLUMN average_lap_time_seconds REAL;

CREATE INDEX idx_race_runs_best_lap
    ON race_runs(best_lap_time_seconds ASC)
    WHERE best_lap_time_seconds IS NOT NULL;
