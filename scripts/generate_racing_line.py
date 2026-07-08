import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from racing_line import RacingLineOptimizer, TorcsTrackMap  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Generate a saved minimum-curvature TORCS racing line."
    )
    parser.add_argument("track_xml", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--measured-length",
        type=float,
        help="TORCS measured lap length used to align distFromStart lookup.",
    )
    parser.add_argument("--spacing", type=float, default=3.0)
    args = parser.parse_args()

    track_map = TorcsTrackMap.from_xml(args.track_xml)
    optimizer = RacingLineOptimizer(
        track_map,
        measured_length=args.measured_length,
        spacing=args.spacing,
    )
    racing_line = optimizer.optimize()
    racing_line.save(args.output)

    metadata = racing_line.payload["optimizer"]
    print(f"Track: {racing_line.track_name}")
    print(f"Waypoints: {len(racing_line.waypoints)}")
    print(f"Length: {racing_line.track_length:.2f} m")
    print(f"Optimizer success: {metadata['success']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
