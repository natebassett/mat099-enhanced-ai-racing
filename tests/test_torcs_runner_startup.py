import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

from runner.torcs_runner import TorcsRunner  # noqa: E402


class TorcsRunnerStartupTests(unittest.TestCase):
    def test_netstat_parser_accepts_expected_torcs_udp_port(self):
        output = """
  UDP    0.0.0.0:3001           *:*                                    1234
"""

        self.assertTrue(TorcsRunner._netstat_reports_server(output, 3001, 1234))

    def test_netstat_parser_accepts_helper_process_when_pid_not_required(self):
        output = """
  UDP    127.0.0.1:3001         *:*                                    9876
"""

        self.assertTrue(TorcsRunner._netstat_reports_server(output, 3001))

    def test_netstat_parser_rejects_wrong_pid_when_pid_required(self):
        output = """
  UDP    127.0.0.1:3001         *:*                                    9876
"""

        self.assertFalse(TorcsRunner._netstat_reports_server(output, 3001, 1234))

    def test_netstat_parser_rejects_other_ports(self):
        output = """
  UDP    127.0.0.1:3002         *:*                                    1234
"""

        self.assertFalse(TorcsRunner._netstat_reports_server(output, 3001, 1234))

    def test_telemetry_emission_interval_keeps_first_sample(self):
        emitted_steps = [
            step
            for step in range(1, 76)
            if TorcsRunner._should_emit_telemetry_sample(step, 25)
        ]

        self.assertEqual(emitted_steps, [1, 25, 50, 75])

    def test_time_based_telemetry_emission_keeps_first_sample(self):
        self.assertTrue(
            TorcsRunner._should_emit_telemetry_sample(
                1,
                now=0.0,
                next_sample_at=10.0,
            )
        )
        self.assertFalse(
            TorcsRunner._should_emit_telemetry_sample(
                20,
                now=9.9,
                next_sample_at=10.0,
            )
        )
        self.assertTrue(
            TorcsRunner._should_emit_telemetry_sample(
                21,
                now=10.0,
                next_sample_at=10.0,
            )
        )

    def test_telemetry_interval_seconds_are_optional_and_non_negative(self):
        self.assertIsNone(TorcsRunner._normalise_telemetry_interval_seconds(None))
        self.assertEqual(TorcsRunner._normalise_telemetry_interval_seconds(1.25), 1.25)
        self.assertEqual(TorcsRunner._normalise_telemetry_interval_seconds(-2), 0.0)

    def test_stuck_watchdog_requires_sustained_low_speed_after_startup(self):
        self.assertTrue(TorcsRunner._is_stuck_speed(0.079))
        self.assertFalse(TorcsRunner._is_stuck_speed(0.08))
        self.assertFalse(TorcsRunner._should_stop_for_stuck(250, 30.0))
        self.assertFalse(TorcsRunner._should_stop_for_stuck(251, 7.9))
        self.assertTrue(TorcsRunner._should_stop_for_stuck(251, 8.0))

    def test_lap_time_parser_treats_bad_values_as_zero(self):
        self.assertEqual(TorcsRunner._lap_time_seconds({"curLapTime": "12.5"}), 12.5)
        self.assertEqual(TorcsRunner._lap_time_seconds({"curLapTime": None}), 0.0)
        self.assertEqual(TorcsRunner._lap_time_seconds({"curLapTime": "bad"}), 0.0)


if __name__ == "__main__":
    unittest.main()
