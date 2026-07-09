def practice_finish_is_plausible(initial_telemetry, final_telemetry, agent):
    """Reject socket closures caused by a stopped/off-track Practice car."""
    initial_distance = float(initial_telemetry.get("distRaced", 0.0) or 0.0)
    final_distance = float(final_telemetry.get("distRaced", 0.0) or 0.0)
    distance_raced = max(0.0, final_distance - initial_distance)
    racing_line = getattr(agent, "racing_line", None)
    expected_length = float(
        getattr(racing_line, "track_length", 0.0)
        or initial_telemetry.get("distFromStart", 0.0)
        or 0.0
    )
    return expected_length > 0.0 and distance_raced >= expected_length * 0.80


class LapTracker:
    """Detect completed laps from TORCS' persistent lastLapTime sensor."""

    def __init__(self, initial_last_lap_time=0.0):
        initial = float(initial_last_lap_time or 0.0)
        self._last_seen = initial if initial > 0 else None
        self.lap_times = []

    def update(self, telemetry):
        lap_time = float(telemetry.get("lastLapTime", 0.0) or 0.0)
        return self.record(lap_time)

    def record(self, lap_time):
        lap_time = float(lap_time or 0.0)
        if lap_time <= 0:
            return None
        if self._last_seen is not None and abs(lap_time - self._last_seen) < 1e-6:
            return None

        self._last_seen = lap_time
        self.lap_times.append(lap_time)
        return lap_time

    @property
    def laps_completed(self):
        return len(self.lap_times)

    @property
    def best_lap_time(self):
        return min(self.lap_times) if self.lap_times else None

    @property
    def average_lap_time(self):
        if not self.lap_times:
            return None
        return sum(self.lap_times) / len(self.lap_times)
