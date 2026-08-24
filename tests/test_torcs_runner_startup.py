import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

from runner import torcs_runner as torcs_runner_module  # noqa: E402
from runner.torcs_runner import TorcsRunner  # noqa: E402


class TorcsRunnerStartupTests(unittest.TestCase):
    def test_launch_retries_after_native_startup_crash(self):
        runner = TorcsRunner()
        runner.torcs_path = Path(__file__)
        failed_process = mock.Mock()
        failed_process.poll.return_value = 3221225477
        healthy_process = mock.Mock()
        healthy_process.poll.return_value = None

        with (
            mock.patch.object(
                torcs_runner_module.subprocess,
                "Popen",
                side_effect=[failed_process, healthy_process],
            ) as popen,
            mock.patch.object(torcs_runner_module.time, "sleep") as sleep,
            mock.patch.object(runner, "_start_practice_race") as start_race,
        ):
            runner.launch()

        self.assertEqual(popen.call_count, 2)
        self.assertEqual(sleep.call_args_list, [mock.call(8.0), mock.call(2.0), mock.call(8.0)])
        self.assertIs(runner.torcs_process, healthy_process)
        start_race.assert_called_once_with()

    def test_launch_reports_exit_codes_after_all_attempts_fail(self):
        runner = TorcsRunner()
        runner.torcs_path = Path(__file__)
        failed_processes = []
        for return_code in (3221225477, 1, 3221225477):
            process = mock.Mock()
            process.poll.return_value = return_code
            failed_processes.append(process)

        with (
            mock.patch.object(
                torcs_runner_module.subprocess,
                "Popen",
                side_effect=failed_processes,
            ),
            mock.patch.object(torcs_runner_module.time, "sleep") as sleep,
            mock.patch.object(runner, "_start_practice_race") as start_race,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "3221225477, 1, 3221225477",
            ):
                runner.launch()

        self.assertEqual(
            sleep.call_args_list,
            [
                mock.call(8.0),
                mock.call(2.0),
                mock.call(8.0),
                mock.call(4.0),
                mock.call(8.0),
            ],
        )
        self.assertIsNone(runner.torcs_process)
        start_race.assert_not_called()

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

    def test_track_sensor_values_are_normalized_for_sampled_telemetry(self):
        sensors = TorcsRunner._track_sensor_values(
            {"track": [str(value) for value in range(25)]}
        )

        self.assertEqual(sensors, [float(value) for value in range(19)])

    def test_recorded_telemetry_includes_optional_agent_debug_fields(self):
        class DebugAgent:
            def telemetry_debug(self):
                return {
                    "dyna_q_state": "1,2,3",
                    "dyna_q_action_label": "straight / maintain",
                    "dyna_q_reward": 1.25,
                }

        runner = TorcsRunner()
        results = {"steps": 42, "telemetry_samples": []}
        runner._record_telemetry_sample(
            results,
            {
                "distFromStart": 100.0,
                "distRaced": 100.0,
                "speedX": 50.0,
                "track": [200.0] * 19,
            },
            {"steer": 0.0, "accel": 0.5, "brake": 0.0, "gear": 2},
            0.5,
            False,
            agent=DebugAgent(),
        )

        sample = results["telemetry_samples"][0]
        self.assertEqual(sample["dyna_q_state"], "1,2,3")
        self.assertEqual(sample["dyna_q_action_label"], "straight / maintain")
        self.assertEqual(sample["dyna_q_reward"], 1.25)

    def test_shutdown_on_finish_flag_can_keep_torcs_process_alive(self):
        runner = TorcsRunner()
        runner.env = FakeFinishedEnv()
        runner.torcs_process = FakeProcess()

        runner.run(FakeFinishedAgent(), shutdown_on_finish=False)

        self.assertIsNotNone(runner.torcs_process)
        self.assertFalse(runner.torcs_process.terminated)

    def test_run_shuts_down_by_default(self):
        runner = TorcsRunner()
        runner.env = FakeFinishedEnv()
        process = FakeProcess()
        runner.torcs_process = process

        runner.run(FakeFinishedAgent())

        self.assertIsNone(runner.torcs_process)
        self.assertTrue(process.terminated)


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class FakeClient:
    def __init__(self):
        self.S = types.SimpleNamespace(
            d={
                "lastLapTime": 0.0,
                "curLapTime": 0.0,
                "speedX": 0.0,
                "angle": 0.0,
                "damage": 0.0,
                "track": [200.0] * 19,
            }
        )
        self.R = types.SimpleNamespace(d={})

    def respond_to_server(self):
        pass


class FakeFinishedEnv:
    default_speed = 50.0

    def __init__(self):
        self.client = FakeClient()

    def reset(self, relaunch=False):
        return types.SimpleNamespace(speedX=0.0, track=[200.0] * 19)

    def end(self):
        pass


class FakeFinishedAgent:
    uses_full_control = True
    max_steps = 1
    target_laps = 1

    def reset(self):
        pass

    def act(self, _observation, _telemetry):
        return {
            "steer": 0.0,
            "accel": 0.0,
            "brake": 1.0,
            "gear": 1,
            "terminate": True,
            "termination_reason": "test_complete",
        }

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
