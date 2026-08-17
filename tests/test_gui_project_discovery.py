from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.project_discovery import (  # noqa: E402
    compatible_tracks_for_agent,
    discover_agents,
    discover_cars,
    discover_run_history,
    discover_tracks,
)


class ProjectDiscoveryTests(unittest.TestCase):
    def test_discovers_real_agent_metadata(self):
        agents = discover_agents()
        agent_types = {agent.agent_type for agent in agents}

        self.assertIn("map_aware", agent_types)
        self.assertIn("rule_based", agent_types)
        self.assertIn("random", agent_types)
        self.assertIn("dyna_q_learning", agent_types)
        self.assertIn("dyna_q_finalised", agent_types)
        self.assertIn("td3_scratch", agent_types)
        self.assertTrue(any(agent.uses_full_control for agent in agents))

    def test_discovers_torcs_tracks_from_xml(self):
        tracks = discover_tracks(PROJECT_ROOT)
        g_track_3 = next(track for track in tracks if track.track_id == "g-track-3")

        self.assertEqual(g_track_3.category, "road")
        self.assertEqual(g_track_3.display_name, "CG track 3")
        self.assertIsNotNone(g_track_3.racing_line_path)
        self.assertGreaterEqual(len(tracks), 20)

    def test_map_aware_agent_only_offers_raceline_ready_tracks(self):
        agents = discover_agents()
        tracks = discover_tracks(PROJECT_ROOT)
        map_aware = next(agent for agent in agents if agent.agent_type == "map_aware")
        compatible = compatible_tracks_for_agent(map_aware, tracks)

        self.assertTrue(map_aware.requires_racing_line)
        self.assertEqual(
            {track.track_id for track in compatible},
            {"corkscrew", "g-track-3"},
        )

    def test_discovers_torcs_cars_from_xml(self):
        cars = discover_cars(PROJECT_ROOT)
        alfa = next(car for car in cars if car.car_id == "155-DTM")

        self.assertEqual(alfa.display_name, "Alfa Romeo 155 DTM")
        self.assertEqual(alfa.category, "Track-4WD-GrB")
        self.assertGreaterEqual(len(cars), 30)

    def test_loads_saved_run_history(self):
        runs, source = discover_run_history(PROJECT_ROOT, limit=5)

        self.assertIn("race_results.db", source)
        self.assertLessEqual(len(runs), 5)
        self.assertTrue(all(run.agent_name for run in runs))


if __name__ == "__main__":
    unittest.main()
