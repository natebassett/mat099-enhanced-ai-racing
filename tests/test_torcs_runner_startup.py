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


if __name__ == "__main__":
    unittest.main()
