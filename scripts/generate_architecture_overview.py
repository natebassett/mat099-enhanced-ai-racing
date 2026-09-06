"""Generate a concise, dissertation-ready architecture overview."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    PROJECT_ROOT / "docs" / "assets" / "architecture-overview",
    PROJECT_ROOT / "dissertation" / "figures" / "architecture-overview",
]

BACKGROUND = "#11151D"
NODE = "#1B222E"
NODE_ALT = "#202A37"
OUTLINE = "#8A97A7"
TEXT = "#F4F7FA"
MUTED = "#C2CCD7"
RUNTIME = "#D7DEE7"
EVIDENCE = "#67C6B5"
OFFLINE = "#E6B76A"


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fill: str = NODE,
) -> None:
    """Draw one restrained labelled node."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.9,
            edgecolor=OUTLINE,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=8.9,
        linespacing=1.25,
        zorder=3,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = RUNTIME,
    dashed: bool = False,
    connection: str = "arc3,rad=0",
) -> None:
    """Draw a directional relation without cluttering the diagram."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            linestyle="dashed" if dashed else "solid",
            color=colour,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def polyline_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    colour: str = RUNTIME,
) -> None:
    """Draw a routed return path with a single arrowhead at its destination."""
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=colour, linewidth=1.0, solid_capstyle="round", zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def edge_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    colour: str = MUTED,
) -> None:
    """Add a small edge label that remains readable against the dark canvas."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=colour,
        fontsize=6.8,
        zorder=4,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": BACKGROUND, "edgecolor": "none"},
    )


def group_label(ax: plt.Axes, x: float, y: float, text: str, colour: str) -> None:
    """Draw a quiet group heading without turning the diagram into a dashboard."""
    ax.text(
        x,
        y,
        text.upper(),
        ha="left",
        va="center",
        color=colour,
        fontsize=7.1,
        fontweight="bold",
    )


def generate() -> None:
    """Create the PNG and PDF used by the documentation and dissertation."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})
    figure, ax = plt.subplots(figsize=(13.6, 7.4))
    figure.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    group_label(ax, 0.55, 9.35, "Offline research workflow", OFFLINE)
    group_label(ax, 4.15, 9.35, "Packaged novice-facing application", RUNTIME)

    # Inputs produced by the offline research workflow, outside the end-user app.
    box(
        ax,
        0.55,
        7.70,
        2.55,
        0.76,
        "Training and batch\nevaluation scripts",
        fill=NODE_ALT,
    )
    box(
        ax,
        0.55,
        5.35,
        2.85,
        0.94,
        "Runtime assets and evidence\npolicies / racing lines /\nevaluation summaries",
        fill=NODE_ALT,
    )

    # The shipped desktop application.
    box(ax, 4.15, 7.70, 2.20, 0.76, "EnhancedAIRacing.exe\nWindows entry point")
    box(
        ax,
        7.05,
        7.45,
        3.15,
        1.26,
        "PySide6 desktop GUI\nDashboard / Results / Review\nCompare / Agent Lab",
    )
    box(ax, 11.20, 7.70, 2.35, 0.76, "RaceWorker\nbackground QThread")

    # Live race execution components.
    box(
        ax,
        4.25,
        4.55,
        2.70,
        0.94,
        "Selected agent\nsrc/agents",
    )
    box(
        ax,
        8.00,
        4.55,
        2.70,
        0.94,
        "TorcsRunner\nand gym_torcs adapter",
    )
    box(
        ax,
        12.00,
        4.55,
        2.75,
        0.94,
        "TORCS simulator\nseparate process",
    )

    # Persisted run evidence, used by the post-race interface.
    box(ax, 8.00, 1.55, 2.70, 0.82, "RaceRepository")
    box(
        ax,
        12.00,
        1.55,
        2.75,
        0.82,
        "SQLite race_results.db\nruns / metrics / telemetry",
        fill=NODE_ALT,
    )

    # Packaging and GUI orchestration.
    arrow(ax, (6.35, 8.08), (7.05, 8.08), colour="#7BADE0")
    arrow(ax, (10.20, 8.08), (11.20, 8.08), colour="#7BADE0")
    edge_label(ax, 10.70, 8.35, "start / stop / selected setup", colour="#9BC7F2")
    arrow(ax, (11.20, 7.82), (10.20, 7.82), colour=EVIDENCE)
    edge_label(ax, 10.70, 7.55, "Qt status and live telemetry", colour=EVIDENCE)

    # The worker creates the selected agent and invokes the runner.
    polyline_arrow(
        ax,
        [(11.20, 8.23), (10.75, 6.25), (6.95, 6.25), (6.95, 5.02)],
        colour="#7BADE0",
    )
    edge_label(ax, 8.85, 6.48, "instantiates selected driver", colour="#9BC7F2")
    polyline_arrow(
        ax,
        [(13.55, 8.08), (14.35, 8.08), (14.35, 6.12), (10.70, 5.02)],
        colour=RUNTIME,
    )
    edge_label(ax, 13.12, 6.38, "launches and runs", colour=MUTED)

    # Runtime inputs and the actual agent-environment exchange.
    arrow(ax, (3.40, 5.82), (4.25, 5.02), colour=OFFLINE, dashed=True)
    edge_label(ax, 3.83, 5.63, "loads", colour=OFFLINE)
    arrow(ax, (8.00, 5.18), (6.95, 5.18))
    edge_label(ax, 7.48, 5.43, "observations")
    arrow(ax, (6.95, 4.83), (8.00, 4.83))
    edge_label(ax, 7.48, 4.56, "control actions")
    arrow(ax, (10.70, 5.02), (12.00, 5.02))
    edge_label(ax, 11.35, 5.30, "TORCS protocol")

    # Live and stored evidence return to the user-facing application.
    arrow(ax, (9.35, 5.49), (11.20, 8.08), colour=EVIDENCE, connection="arc3,rad=0.04")
    edge_label(ax, 10.62, 6.77, "telemetry callback", colour=EVIDENCE)
    polyline_arrow(
        ax,
        [(13.55, 7.95), (14.90, 7.95), (14.90, 3.05), (10.70, 2.05)],
        colour=EVIDENCE,
    )
    edge_label(ax, 13.95, 3.26, "completed run + samples", colour=EVIDENCE)
    arrow(ax, (10.70, 1.96), (12.00, 1.96), colour=EVIDENCE)
    polyline_arrow(
        ax,
        [(8.00, 1.96), (7.35, 1.96), (7.35, 7.45), (7.62, 7.45)],
        colour=EVIDENCE,
    )
    edge_label(ax, 6.70, 3.15, "saved runs for review and comparison", colour=EVIDENCE)

    # Offline scripts create the assets and summaries that are shipped read-only.
    arrow(ax, (1.82, 7.70), (1.82, 6.29), colour=OFFLINE, dashed=True)
    edge_label(ax, 2.50, 6.94, "produces", colour=OFFLINE)

    figure.tight_layout(pad=0.25)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output.with_suffix(".png"), bbox_inches="tight", facecolor=BACKGROUND)
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=BACKGROUND)
        print(output.with_suffix(".png"))
        print(output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    generate()
