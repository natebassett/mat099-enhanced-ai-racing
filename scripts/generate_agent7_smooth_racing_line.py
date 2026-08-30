import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from racing_line.smoothing import build_smoothed_racing_line  # noqa: E402


DEFAULT_BASE_LINE = PROJECT_ROOT / "data" / "racing_lines" / "g-track-3.json"
DEFAULT_TRACK_XML = (
    PROJECT_ROOT / "torcs" / "tracks" / "road" / "g-track-3" / "g-track-3.xml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "racing_lines"
    / "g-track-3-agent7-smooth-v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a smoother Agent 7 racing line without replacing the base line."
    )
    parser.add_argument("--base-line", type=Path, default=DEFAULT_BASE_LINE)
    parser.add_argument("--track-xml", type=Path, default=DEFAULT_TRACK_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regularization", type=float, default=30.0)
    parser.add_argument("--curvature-window", type=int, default=21)
    args = parser.parse_args()

    for label, path in (("base line", args.base_line), ("track XML", args.track_xml)):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.output.resolve() == args.base_line.resolve():
        parser.error("--output must not replace --base-line")

    line = build_smoothed_racing_line(
        args.base_line,
        args.track_xml,
        regularization=args.regularization,
        curvature_window=args.curvature_window,
    )
    line.save(args.output)
    before = line.payload["optimizer"]["quality_before"]
    after = line.payload["optimizer"]["quality_after"]
    print(f"Saved improved racing line: {args.output}")
    print(f"Waypoints: {len(line.waypoints):,}")
    print(
        "Target second-difference RMS: "
        f"{before['target_second_difference_rms']:.6f} -> "
        f"{after['target_second_difference_rms']:.6f}"
    )
    print(
        "Curvature-step RMS: "
        f"{before['curvature_step_rms']:.6f} -> "
        f"{after['curvature_step_rms']:.6f}"
    )
    print(
        "Maximum curvature step: "
        f"{before['maximum_curvature_step']:.6f} -> "
        f"{after['maximum_curvature_step']:.6f}"
    )


if __name__ == "__main__":
    main()
