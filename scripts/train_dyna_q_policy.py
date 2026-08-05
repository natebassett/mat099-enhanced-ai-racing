from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.dyna_q_agent import (  # noqa: E402
    DEFAULT_FINAL_POLICY_PATH,
    DEFAULT_LATEST_POLICY_PATH,
    DynaQLearningAgent,
)
from gui.torcs_config import TorcsRaceSetup, TorcsRuntimeConfig  # noqa: E402
from runner.torcs_runner import TorcsRunner  # noqa: E402


DEFAULT_TRACK_ID = "g-track-3"
DEFAULT_TRACK_CATEGORY = "road"
DEFAULT_CAR_ID = "car1-ow1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the from-scratch Dyna-Q policy in TORCS and save the latest "
            "Q-table for the finalised agent."
        )
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--track", default=DEFAULT_TRACK_ID)
    parser.add_argument("--track-category", default=DEFAULT_TRACK_CATEGORY)
    parser.add_argument("--car", default=DEFAULT_CAR_ID)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_LATEST_POLICY_PATH)
    parser.add_argument("--promote-final", action="store_true")
    args = parser.parse_args()

    best_lap = None
    for episode in range(1, max(1, args.episodes) + 1):
        print(f"\n=== Dyna-Q training episode {episode}/{args.episodes} ===")
        runner = TorcsRunner()
        agent = DynaQLearningAgent(policy_path=args.policy_path, load_existing=True)
        setup = TorcsRaceSetup(
            track_id=args.track,
            track_category=args.track_category,
            car_id=args.car,
        )

        with TorcsRuntimeConfig(setup):
            runner.launch()
            runner.connect()
            runner.load_track(args.track)
            results = runner.run(agent)

        lap_time = results.get("best_lap_time_seconds")
        if isinstance(lap_time, (int, float)):
            best_lap = lap_time if best_lap is None else min(best_lap, float(lap_time))
        print(
            "Episode result: "
            f"reason={results.get('termination_reason')} | "
            f"steps={results.get('steps')} | "
            f"laps={results.get('laps_completed')} | "
            f"best_lap={lap_time}"
        )

    if args.promote_final and args.policy_path.is_file():
        DEFAULT_FINAL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.policy_path, DEFAULT_FINAL_POLICY_PATH)
        print(f"Promoted latest policy to {DEFAULT_FINAL_POLICY_PATH}")

    if best_lap is not None:
        print(f"Best completed lap during this training run: {best_lap:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
