"""Plot actual track position against the map-aware raceline targets."""

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DATA = PROJECT_ROOT / "data" / "generated"
DEFAULT_INPUT = GENERATED_DATA / "map_aware_telemetry.csv"
DEFAULT_OUTPUT = GENERATED_DATA / "map_aware_tracking.svg"


SERIES = [
    ("actualTrackPos", "actual", "#1f77b4", "solid"),
    ("trackPos", "actual", "#1f77b4", "solid"),
    ("rawRacingLinePos", "raw line", "#9467bd", "dash"),
    ("previewTrackPos", "preview", "#d62728", "dash"),
    ("practicalTrackPos", "practical", "#ff7f0e", "dash"),
    ("targetTrackPos", "target", "#2ca02c", "solid"),
]


def parse_float(value, default=0.0):
    if value in (None, ""):
        return default
    return float(value)


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def downsample(rows, maximum=1400):
    if len(rows) <= maximum:
        return rows
    step = math.ceil(len(rows) / maximum)
    return rows[::step]


def polyline(rows, column, x_min, x_span, width, height, margin):
    points = []
    for row in rows:
        if row.get(column) in (None, ""):
            continue
        x = margin + (parse_float(row["distFromStart"]) - x_min) / x_span * width
        y_value = max(-1.15, min(1.15, parse_float(row[column])))
        y = margin + (1.15 - y_value) / 2.3 * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_svg(rows):
    rows = downsample(rows)
    distances = [parse_float(row["distFromStart"]) for row in rows]
    x_min = min(distances)
    x_max = max(distances)
    x_span = max(x_max - x_min, 1.0)

    outer_width = 1100
    outer_height = 520
    margin = 58
    width = outer_width - margin * 2
    height = outer_height - margin * 2

    grid_lines = []
    for value in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        y = margin + (1.15 - value) / 2.3 * height
        grid_lines.append(
            f'<line x1="{margin}" y1="{y:.1f}" x2="{margin + width}" '
            f'y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{margin - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#555">{value:.1f}</text>'
        )

    path_lines = []
    legend = []
    for index, (column, label, color, style) in enumerate(SERIES):
        if column not in rows[0]:
            continue
        points = polyline(rows, column, x_min, x_span, width, height, margin)
        if not points:
            continue
        dash = ' stroke-dasharray="8 5"' if style == "dash" else ""
        path_lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2.4"{dash}/>'
        )
        lx = margin + index * 155
        ly = 28
        legend.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2.4"{dash}/>'
        )
        legend.append(
            f'<text x="{lx + 36}" y="{ly + 4}" font-size="13" fill="#333">'
            f'{label}</text>'
        )

    x_axis_y = margin + height
    labels = [
        f'<text x="{outer_width / 2:.1f}" y="{outer_height - 16}" '
        f'text-anchor="middle" font-size="13" fill="#333">distFromStart (m)</text>',
        f'<text x="18" y="{outer_height / 2:.1f}" transform="rotate(-90 18 '
        f'{outer_height / 2:.1f})" text-anchor="middle" font-size="13" '
        f'fill="#333">trackPos</text>',
        f'<text x="{margin}" y="{x_axis_y + 22}" font-size="12" fill="#555">'
        f'{x_min:.0f} m</text>',
        f'<text x="{margin + width}" y="{x_axis_y + 22}" text-anchor="end" '
        f'font-size="12" fill="#555">{x_max:.0f} m</text>',
    ]

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{outer_width}" '
            f'height="{outer_height}" viewBox="0 0 {outer_width} {outer_height}" '
            'role="img" aria-label="Raceline tracking plot">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            *legend,
            *grid_lines,
            f'<line x1="{margin}" y1="{x_axis_y}" x2="{margin + width}" '
            f'y2="{x_axis_y}" stroke="#888" stroke-width="1.2"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{x_axis_y}" '
            f'stroke="#888" stroke-width="1.2"/>',
            *path_lines,
            *labels,
            "</svg>",
        ]
    )


def plot(input_path, output_path):
    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No telemetry rows found in {input_path}")
    svg = render_svg(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(plot(args.input, args.output))


if __name__ == "__main__":
    main()
