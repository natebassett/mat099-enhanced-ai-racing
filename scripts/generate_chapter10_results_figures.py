"""Generate dissertation figures for Chapter 10 results.

The figures are built from the local race-results database and the repeated
evaluation JSON files so the dissertation visuals remain reproducible.
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DB = PROJECT_ROOT / "data" / "generated" / "race_results.db"
EVALUATION_ROOT = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "chapter10"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"
DARK = "#222222"
LIGHT_GREY = "#E8E8E8"


@dataclass
class ResultRow:
    label: str
    source: str
    runs: int
    completed: int
    completion_rate: float
    best_lap: float | None
    median_lap: float | None
    off_track_events: int | None = None


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": DARK,
            "axes.titlecolor": DARK,
            "axes.grid": True,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


def db_group(label: str, source: str, where_clause: str) -> ResultRow:
    with sqlite3.connect(RESULTS_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT best_lap_time_seconds, laps_completed, off_track_count
            FROM race_runs
            WHERE track = 'g-track-3' AND ({where_clause})
            """
        ).fetchall()

    completed_rows = [row for row in rows if row["laps_completed"] > 0]
    lap_times = [
        row["best_lap_time_seconds"]
        for row in completed_rows
        if row["best_lap_time_seconds"] is not None
    ]
    runs = len(rows)
    completed = len(completed_rows)
    return ResultRow(
        label=label,
        source=source,
        runs=runs,
        completed=completed,
        completion_rate=completed / runs if runs else 0.0,
        best_lap=min(lap_times) if lap_times else None,
        median_lap=statistics.median(lap_times) if lap_times else None,
        off_track_events=sum(row["off_track_count"] for row in rows),
    )


def load_evaluation(path: Path, label: str, source: str) -> ResultRow:
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if episodes:
        runs = len(episodes)
        completed = sum(1 for ep in episodes if ep.get("laps_completed", 0) > 0)
        lap_times = [
            ep.get("best_lap_time_seconds")
            for ep in episodes
            if ep.get("best_lap_time_seconds") is not None
        ]
        off_track_events = sum(ep.get("off_track_steps", 0) for ep in episodes)
    else:
        runs = int(data.get("repeats", 0))
        completed = int(data.get("completed_laps", 0))
        lap_times = []
        off_track_events = None

    return ResultRow(
        label=label,
        source=source,
        runs=runs,
        completed=completed,
        completion_rate=completed / runs if runs else data.get("completion_rate", 0.0),
        best_lap=min(lap_times) if lap_times else data.get("fastest_lap_time_seconds"),
        median_lap=statistics.median(lap_times)
        if lap_times
        else data.get("median_lap_time_seconds"),
        off_track_events=off_track_events,
    )


def combine_evaluations(label: str, source: str, paths: list[Path]) -> ResultRow:
    rows = [load_evaluation(path, label, source) for path in paths]
    lap_times: list[float] = []
    completed = 0
    runs = 0
    off_track_events = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        runs += len(episodes) if episodes else int(data.get("repeats", 0))
        completed += sum(1 for ep in episodes if ep.get("laps_completed", 0) > 0)
        lap_times.extend(
            ep.get("best_lap_time_seconds")
            for ep in episodes
            if ep.get("best_lap_time_seconds") is not None
        )
        off_track_events += sum(ep.get("off_track_steps", 0) for ep in episodes)
    return ResultRow(
        label=label,
        source=source,
        runs=runs,
        completed=completed,
        completion_rate=completed / runs if runs else 0.0,
        best_lap=min(lap_times) if lap_times else min(
            row.best_lap for row in rows if row.best_lap is not None
        ),
        median_lap=statistics.median(lap_times) if lap_times else None,
        off_track_events=off_track_events,
    )


def main_results() -> list[ResultRow]:
    agent7_v3 = EVALUATION_ROOT / "agent7_n_step_td3_v3" / "n_step_td3_evaluation_20260831-000440.json"
    agent7_v4 = EVALUATION_ROOT / "agent7_n_step_td3_v4" / "n_step_td3_evaluation_20260830-233142.json"
    agent8_paths = [
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-131845.json",
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-132634.json",
    ]
    return [
        db_group("Rule-based anti-spin", "application DB", "agent_id IN (5, 47)"),
        db_group("Map-aware racing line", "application DB", "agent_id = 46"),
        db_group("Dyna-Q learning", "application DB", "agent_id = 50"),
        db_group("TD3 scratch", "application DB", "agent_id IN (51, 52)"),
        load_evaluation(agent7_v3, "Agent 7 TD3 v3", "20-run evaluation"),
        load_evaluation(agent7_v4, "Agent 7 TD3 v4", "10-run evaluation"),
        combine_evaluations("Agent 8 sensor-only TD3", "two 20-run evaluations", agent8_paths),
    ]


def plot_completion(results: list[ResultRow]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    labels = [row.label for row in results]
    values = [row.completion_rate for row in results]
    colors = [GREY, GREEN, ORANGE, RED, BLUE, BLUE, PURPLE]
    y_pos = list(range(len(results)))

    ax.barh(y_pos, values, color=colors, alpha=0.9)
    ax.set_yticks(y_pos, labels)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Completion rate")
    ax.set_title("Figure 10.1: Lap-completion reliability on g-track-3")
    ax.axvline(0.9, color=DARK, linestyle="--", linewidth=1.2)
    ax.text(0.89, -0.55, "90% reliability reference", ha="right", va="center", fontsize=9)
    for y, row in zip(y_pos, results):
        ax.text(
            min(row.completion_rate + 0.025, 1.02),
            y,
            f"{row.completed}/{row.runs}",
            va="center",
            fontsize=9,
            color=DARK,
        )
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "figure_10_1_completion_reliability")
    plt.close(fig)


def plot_lap_times(results: list[ResultRow]) -> None:
    rows = [row for row in results if row.median_lap is not None]
    rows = sorted(rows, key=lambda row: row.median_lap or 999)
    y_pos = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    medians = [row.median_lap for row in rows]
    bests = [row.best_lap for row in rows]
    labels = [row.label for row in rows]

    ax.hlines(y_pos, bests, medians, color=LIGHT_GREY, linewidth=7)
    ax.scatter(medians, y_pos, color=BLUE, s=90, label="Median completed lap")
    ax.scatter(bests, y_pos, color=ORANGE, marker="D", s=60, label="Fastest completed lap")
    for y, row in zip(y_pos, rows):
        text_x = (row.median_lap or 0) + 3
        ha = "left"
        if row.median_lap and row.median_lap > 260:
            text_x = row.median_lap - 6
            ha = "right"
        ax.text(text_x, y, f"{row.median_lap:.3f}s", va="center", ha=ha, fontsize=9)

    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("Lap time (seconds, lower is better)")
    ax.set_title("Figure 10.2: Completed-lap pace comparison")
    ax.set_xlim(70, max(medians) + 35)
    ax.legend(loc="upper right")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "figure_10_2_lap_time_comparison")
    plt.close(fig)


def plot_speed_reliability(results: list[ResultRow]) -> None:
    rows = [row for row in results if row.median_lap is not None]
    fig, ax = plt.subplots(figsize=(10.4, 6.1))
    point_colors = [PURPLE, GREEN, GREY, BLUE, BLUE, ORANGE]
    for row, color in zip(rows, point_colors):
        size = 70 + row.runs * 11
        ax.scatter(
            row.median_lap,
            row.completion_rate,
            s=size,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
            label=f"{row.label} ({row.completed}/{row.runs})",
        )

    ax.axhline(0.9, color=DARK, linestyle="--", linewidth=1)
    ax.axvline(92.056, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(93.5, 0.055, "Map-aware median", fontsize=8.5, color=GREY)
    ax.set_xlim(75, 325)
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Median completed-lap time (seconds, lower is better)")
    ax.set_ylabel("Completion rate")
    ax.set_title("Figure 10.3: Reliability and speed trade-off")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    save_figure(fig, "figure_10_3_reliability_speed_tradeoff")
    plt.close(fig)


def db_lap_times(where_clause: str) -> list[float]:
    with sqlite3.connect(RESULTS_DB) as connection:
        rows = connection.execute(
            f"""
            SELECT best_lap_time_seconds
            FROM race_runs
            WHERE track = 'g-track-3'
              AND best_lap_time_seconds IS NOT NULL
              AND ({where_clause})
            """
        ).fetchall()
    return [float(row[0]) for row in rows]


def evaluation_lap_times(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        float(ep["best_lap_time_seconds"])
        for ep in data.get("episodes", [])
        if ep.get("best_lap_time_seconds") is not None
    ]


def combined_evaluation_lap_times(paths: list[Path]) -> list[float]:
    laps: list[float] = []
    for path in paths:
        laps.extend(evaluation_lap_times(path))
    return laps


def plot_lap_time_distribution() -> None:
    agent7_v3 = EVALUATION_ROOT / "agent7_n_step_td3_v3" / "n_step_td3_evaluation_20260831-000440.json"
    agent7_v4 = EVALUATION_ROOT / "agent7_n_step_td3_v4" / "n_step_td3_evaluation_20260830-233142.json"
    agent8_paths = [
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-131845.json",
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-132634.json",
    ]
    series = [
        ("Map-aware\nDB runs", db_lap_times("agent_id = 46"), GREEN),
        ("Agent 7 v3\n20-run eval", evaluation_lap_times(agent7_v3), BLUE),
        ("Agent 7 v4\n10-run eval", evaluation_lap_times(agent7_v4), BLUE),
        ("Agent 8\n40-run eval", combined_evaluation_lap_times(agent8_paths), PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    data = [values for _, values, _ in series]
    positions = list(range(1, len(series) + 1))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.8},
        whiskerprops={"color": GREY},
        capprops={"color": GREY},
    )
    for patch, (_, _, color) in zip(box["boxes"], series):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)

    rng = random.Random(42)
    for x, (_, values, color) in zip(positions, series):
        jittered = [x + rng.uniform(-0.09, 0.09) for _ in values]
        ax.scatter(jittered, values, s=26, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        median = statistics.median(values)
        ax.text(x, median - 1.25, f"{median:.3f}s", ha="center", va="top", fontsize=8.5)

    ax.set_xticks(positions, [label for label, _, _ in series])
    ax.set_ylabel("Completed-lap time (seconds)")
    ax.set_title("Figure 10.6: Distribution of completed-lap times")
    ax.set_ylim(78, 112)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "figure_10_6_lap_time_distribution")
    plt.close(fig)


def evaluation_points() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    roots = [
        EVALUATION_ROOT / "agent7_n_step_td3",
        EVALUATION_ROOT / "agent7_n_step_td3_v3",
        EVALUATION_ROOT / "agent7_n_step_td3_v4",
        EVALUATION_ROOT / "agent8_sensor_n_step_td3",
        EVALUATION_ROOT / "agent8_sensor_n_step_td3_v1_continuation",
        EVALUATION_ROOT / "agent8_sensor_n_step_td3_v1_robust_pace",
        EVALUATION_ROOT / "agent8_sensor_n_step_td3_v1_stability",
        EVALUATION_ROOT / "agent8_sensor_n_step_td3_self_imitation_stability",
        EVALUATION_ROOT / "agent8_sensor_n_step_td3_v2_steer070",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            match = re.search(r"(\d{8}-\d{6})", path.name)
            if not match:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            label = "Agent 7" if "agent7" in root.name else "Agent 8"
            points.append(
                {
                    "timestamp": match.group(1),
                    "agent": label,
                    "completion_rate": float(data.get("completion_rate", 0.0)),
                    "median_lap": data.get("median_lap_time_seconds"),
                    "completed_laps": data.get("completed_laps"),
                    "repeats": data.get("repeats"),
                }
            )
    return sorted(points, key=lambda item: str(item["timestamp"]))


def plot_learning_progression() -> None:
    points = evaluation_points()
    agent_points = {
        "Agent 7": [p for p in points if p["agent"] == "Agent 7"],
        "Agent 8": [p for p in points if p["agent"] == "Agent 8"],
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
    palette = {"Agent 7": BLUE, "Agent 8": PURPLE}

    for agent, rows in agent_points.items():
        xs = list(range(1, len(rows) + 1))
        axes[0].plot(
            xs,
            [row["completion_rate"] for row in rows],
            marker="o",
            linewidth=2,
            markersize=4,
            label=agent,
            color=palette[agent],
        )
        median_rows = [(x, row) for x, row in zip(xs, rows) if row["median_lap"] is not None]
        if median_rows:
            axes[1].plot(
                [x for x, _ in median_rows],
                [row["median_lap"] for _, row in median_rows],
                marker="o",
                linewidth=2,
                markersize=4,
                label=agent,
                color=palette[agent],
            )

    axes[0].set_ylabel("Completion rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0.9, color=DARK, linestyle="--", linewidth=1)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Figure 10.4: Evaluation progression for learned TD3 agents")

    axes[1].set_ylabel("Median lap time (s)")
    axes[1].set_xlabel("Evaluation checkpoint order within each agent family")
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y")
    fig.tight_layout()
    save_figure(fig, "figure_10_4_td3_evaluation_progression")
    plt.close(fig)


def plot_outcome_batches() -> None:
    batch_paths = [
        (
            "Agent 7 v3\n20 runs",
            EVALUATION_ROOT
            / "agent7_n_step_td3_v3"
            / "n_step_td3_evaluation_20260831-000440.json",
        ),
        (
            "Agent 7 v4\n10 runs",
            EVALUATION_ROOT
            / "agent7_n_step_td3_v4"
            / "n_step_td3_evaluation_20260830-233142.json",
        ),
        (
            "Agent 8 batch A\n20 runs",
            EVALUATION_ROOT
            / "agent8_sensor_n_step_td3_self_imitation_stability"
            / "n_step_td3_evaluation_20260902-131845.json",
        ),
        (
            "Agent 8 batch B\n20 runs",
            EVALUATION_ROOT
            / "agent8_sensor_n_step_td3_self_imitation_stability"
            / "n_step_td3_evaluation_20260902-132634.json",
        ),
    ]

    labels = []
    completed = []
    failed = []
    for label, path in batch_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = int(data.get("repeats", 0))
        done = int(data.get("completed_laps", 0))
        labels.append(label)
        completed.append(done)
        failed.append(runs - done)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    x_pos = list(range(len(labels)))
    ax.bar(x_pos, completed, color=GREEN, label="Completed")
    ax.bar(x_pos, failed, bottom=completed, color=RED, label="Not completed")
    for x, done, miss in zip(x_pos, completed, failed):
        total = done + miss
        ax.text(x, total + 0.45, f"{done}/{total}", ha="center", fontsize=10)
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel("Evaluation runs")
    ax.set_title("Figure 10.5: Repeated-evaluation outcome composition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_ylim(0, 24)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "figure_10_5_repeated_evaluation_outcomes")
    plt.close(fig)


def agent_color(label: str) -> str:
    """Use one colour per agent family across every Chapter 10 figure."""
    if "Map-aware" in label:
        return GREEN
    if "Rule-based" in label:
        return GREY
    if "Dyna-Q" in label:
        return ORANGE
    if "scratch" in label.lower():
        return RED
    if "Agent 8" in label:
        return PURPLE
    return BLUE


def wilson_interval(completed: int, runs: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if runs <= 0:
        return 0.0, 0.0
    proportion = completed / runs
    denominator = 1.0 + (z * z / runs)
    centre = (proportion + z * z / (2.0 * runs)) / denominator
    half_width = (
        z
        * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * runs)) / runs)
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def plot_completion_with_uncertainty(results: list[ResultRow]) -> None:
    """Show observed completion with uncertainty, retaining each raw denominator."""
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ordered = list(reversed(results))

    for y, row in enumerate(ordered):
        lower, upper = wilson_interval(row.completed, row.runs)
        colour = agent_color(row.label)
        ax.errorbar(
            row.completion_rate,
            y,
            xerr=[
                [max(0.0, row.completion_rate - lower)],
                [max(0.0, upper - row.completion_rate)],
            ],
            fmt="o",
            markersize=8,
            color=colour,
            ecolor=colour,
            elinewidth=2.2,
            capsize=4,
            zorder=3,
        )
        ax.text(
            min(1.015, upper + 0.025),
            y,
            f"{row.completed}/{row.runs}",
            va="center",
            ha="left",
            fontsize=9,
            color=DARK,
        )

    ax.set_yticks(range(len(ordered)), [row.label for row in ordered])
    ax.set_xlim(-0.02, 1.12)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Observed lap-completion proportion (95% Wilson interval)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "figure_10_1_completion_reliability")
    plt.close(fig)


def plot_completed_lap_distributions() -> None:
    """Plot all completed laps, keeping sample-level variation visible."""
    agent7_v3 = EVALUATION_ROOT / "agent7_n_step_td3_v3" / "n_step_td3_evaluation_20260831-000440.json"
    agent7_v4 = EVALUATION_ROOT / "agent7_n_step_td3_v4" / "n_step_td3_evaluation_20260830-233142.json"
    agent8_paths = [
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-131845.json",
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-132634.json",
    ]
    series = [
        ("Map-aware\napplication runs", db_lap_times("agent_id = 46"), GREEN),
        ("Agent 7 v3\n20-run evaluation", evaluation_lap_times(agent7_v3), BLUE),
        ("Agent 7 v4\n10-run evaluation", evaluation_lap_times(agent7_v4), BLUE),
        ("Agent 8\n40-run evaluation", combined_evaluation_lap_times(agent8_paths), PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    values = [lap_times for _, lap_times, _ in series]
    positions = np.arange(1, len(series) + 1)

    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, (_, _, colour) in zip(violins["bodies"], series):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.18)

    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.8},
        whiskerprops={"color": GREY, "linewidth": 1.1},
        capprops={"color": GREY, "linewidth": 1.1},
    )
    for patch, (_, _, colour) in zip(boxes["boxes"], series):
        patch.set_facecolor("white")
        patch.set_edgecolor(colour)
        patch.set_linewidth(1.5)

    rng = random.Random(42)
    for x, (_, lap_times, colour) in zip(positions, series):
        jittered = [x + rng.uniform(-0.11, 0.11) for _ in lap_times]
        ax.scatter(
            jittered,
            lap_times,
            s=30,
            color=colour,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        median = statistics.median(lap_times)
        ax.text(x, median - 0.9, f"median {median:.3f}s\n(n={len(lap_times)})", ha="center", va="top", fontsize=8.2)

    ax.set_xticks(positions, [label for label, _, _ in series])
    ax.set_ylabel("Completed-lap time (seconds)")
    ax.set_ylim(80, 110)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "figure_10_2_lap_time_distribution")
    plt.close(fig)


def plot_primary_performance_map(results: list[ResultRow]) -> None:
    """Map the repeated primary batches in pace-reliability space.

    The figure deliberately excludes the two- and three-run early baselines.
    They remain in the full reliability plot and results table, while this view
    focuses on batches with at least ten attempts and more than one completed lap.
    """
    selected = [
        row
        for row in results
        if row.runs >= 10 and row.completed > 1 and row.median_lap is not None
    ]
    labels = {
        "Map-aware racing line": "Map-aware\nracing line",
        "Agent 7 TD3 v3": "Agent 7\nv3",
        "Agent 7 TD3 v4": "Agent 7\nv4",
        "Agent 8 sensor-only TD3": "Agent 8\nsensor-only",
    }
    offsets = {
        "Map-aware racing line": (-8.0, 0.010),
        "Agent 7 TD3 v3": (1.2, -0.027),
        "Agent 7 TD3 v4": (1.2, 0.012),
        "Agent 8 sensor-only TD3": (1.1, -0.033),
    }
    markers = {
        "Map-aware racing line": "s",
        "Agent 7 TD3 v3": "^",
        "Agent 7 TD3 v4": "o",
        "Agent 8 sensor-only TD3": "P",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for row in selected:
        lower, upper = wilson_interval(row.completed, row.runs)
        colour = agent_color(row.label)
        ax.errorbar(
            row.median_lap,
            row.completion_rate,
            yerr=[
                [max(0.0, row.completion_rate - lower)],
                [max(0.0, upper - row.completion_rate)],
            ],
            fmt=markers[row.label],
            markersize=9,
            color=colour,
            ecolor=colour,
            elinewidth=1.8,
            capsize=3,
            zorder=3,
        )
        dx, dy = offsets[row.label]
        ax.annotate(
            f"{labels[row.label]}\n{row.completed}/{row.runs}",
            xy=(row.median_lap, row.completion_rate),
            xytext=(row.median_lap + dx, row.completion_rate + dy),
            ha="right" if dx < 0 else "left",
            va="center",
            fontsize=8.6,
            color=DARK,
        )

    ax.set_xlim(80, 110)
    ax.set_ylim(0.65, 1.035)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Median completed-lap time (seconds; lower is better)")
    ax.set_ylabel("Observed completion proportion (95% Wilson interval)")
    ax.grid()
    fig.tight_layout()
    save_figure(fig, "figure_10_3_performance_map")
    plt.close(fig)


def plot_agent8_development_ladder() -> None:
    """Show material Agent 8 design milestones without presenting a pseudo learning curve."""
    def evaluation(folder: str, filename: str, label: str) -> ResultRow:
        return load_evaluation(EVALUATION_ROOT / folder / filename, label, "development evaluation")

    champion_paths = [
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-131845.json",
        EVALUATION_ROOT
        / "agent8_sensor_n_step_td3_self_imitation_stability"
        / "n_step_td3_evaluation_20260902-132634.json",
    ]
    milestones = [
        evaluation(
            "agent8_sensor_n_step_td3",
            "n_step_td3_evaluation_20260901-112751.json",
            "v1 base",
        ),
        evaluation(
            "agent8_sensor_n_step_td3",
            "n_step_td3_evaluation_20260901-143256.json",
            "v1 first completion",
        ),
        evaluation(
            "agent8_sensor_n_step_td3",
            "n_step_td3_evaluation_20260901-201320.json",
            "v1 pace candidate",
        ),
        evaluation(
            "agent8_sensor_n_step_td3_v1_stability",
            "n_step_td3_evaluation_20260901-212526.json",
            "v1 stability candidate",
        ),
        evaluation(
            "agent8_sensor_n_step_td3_v1_robust_pace",
            "n_step_td3_evaluation_20260902-005424.json",
            "v1 robust-pace trial",
        ),
        evaluation(
            "agent8_sensor_n_step_td3_v2_steer070",
            "n_step_td3_evaluation_20260901-180954.json",
            "v2 steer-limit trial",
        ),
        evaluation(
            "agent8_sensor_n_step_td3_self_imitation_stability",
            "n_step_td3_evaluation_20260902-122432.json",
            "v2 self-imitation candidate",
        ),
        combine_evaluations(
            "v2 self-imitation champion",
            "two repeated evaluations",
            champion_paths,
        ),
    ]
    tick_labels = [
        "v1\nbase",
        "v1\nfirst lap",
        "v1\npace",
        "v1\nstability",
        "v1\nrobust pace",
        "v2\nsteer limit",
        "v2\nself-imitation\ncandidate",
        "v2\nself-imitation\nchampion",
    ]
    positions = np.arange(1, len(milestones) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.7), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]})

    for position, row in zip(positions, milestones):
        lower, upper = wilson_interval(row.completed, row.runs)
        final = row.label.endswith("champion")
        colour = PURPLE if final else GREY
        axes[0].errorbar(
            position,
            row.completion_rate,
            yerr=[
                [max(0.0, row.completion_rate - lower)],
                [max(0.0, upper - row.completion_rate)],
            ],
            fmt="o",
            markersize=9 if final else 7,
            color=colour,
            ecolor=colour,
            elinewidth=2.0,
            capsize=3,
            zorder=3,
        )
        axes[0].text(position, min(1.045, upper + 0.07), f"{row.completed}/{row.runs}", ha="center", va="bottom", fontsize=8.4)
        if row.median_lap is not None:
            axes[1].scatter(position, row.median_lap, s=75 if final else 54, color=colour, zorder=3)
            axes[1].text(position, row.median_lap + 4.0, f"{row.median_lap:.3f}s", ha="center", va="bottom", fontsize=8.2)
        else:
            axes[1].plot(position, 78.5, marker="x", markersize=7, markeredgewidth=1.6, color=GREY)

    axes[0].set_ylim(-0.06, 1.08)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Completion proportion\n(95% Wilson interval)")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)
    axes[1].set_ylim(75, 152)
    axes[1].set_ylabel("Median completed-lap\ntime (seconds)")
    axes[1].set_xticks(positions, tick_labels, fontsize=8.6)
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)
    axes[1].annotate(
        "x = no completed lap",
        xy=(1, 78.5),
        xytext=(1.55, 89),
        arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.8},
        fontsize=8.3,
        color=GREY,
    )
    fig.tight_layout()
    save_figure(fig, "figure_10_4_agent8_development")
    plt.close(fig)


def run_telemetry(run_id: int) -> list[sqlite3.Row]:
    with sqlite3.connect(RESULTS_DB) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT dist_from_start, speed_x, track_pos, steer, accel, brake
            FROM run_telemetry
            WHERE run_id = ?
            ORDER BY step
            """,
            (run_id,),
        ).fetchall()


def profile_on_progress_grid(rows: list[sqlite3.Row], column: str, grid: np.ndarray) -> np.ndarray:
    distances = np.asarray([float(row["dist_from_start"]) for row in rows], dtype=float)
    values = np.asarray([float(row[column]) for row in rows], dtype=float)
    valid = np.isfinite(distances) & np.isfinite(values)
    distances = distances[valid]
    values = values[valid]
    progress = 100.0 * (distances - distances.min()) / (distances.max() - distances.min())
    order = np.argsort(progress)
    progress = progress[order]
    values = values[order]
    unique_progress, unique_indices = np.unique(progress, return_index=True)
    return np.interp(grid, unique_progress, values[unique_indices])


def smooth_profile(trace: np.ndarray, window: int = 9) -> np.ndarray:
    """Apply a short, edge-preserving moving average for display only."""
    padding = window // 2
    padded = np.pad(trace, (padding, padding), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_representative_control_profiles() -> None:
    """Show descriptive telemetry profiles for median-like completed laps.

    These traces are explanatory rather than inferential: each is one completed
    run chosen near the stored median for its agent/configuration.
    """
    profiles = [
        ("Map-aware racing line\n92.056s", 80, GREEN),
        ("Agent 7 TD3 v3\n104.180s", 104, BLUE),
        ("Agent 8 sensor-only TD3\n83.038s", 116, PURPLE),
    ]
    grid = np.linspace(0.0, 100.0, 300)
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex=True)
    axes = axes.ravel()
    panels = [
        ("speed_x", "Speed (km/h)", "Speed"),
        ("track_pos", "Lateral track position", "Track position"),
        ("steer", "Steering command", "Steering"),
        ("longitudinal", "Longitudinal command", "Throttle and brake"),
    ]

    for label, run_id, colour in profiles:
        rows = run_telemetry(run_id)
        if not rows:
            raise RuntimeError(f"No stored telemetry found for representative run {run_id}.")
        for ax, (column, ylabel, _) in zip(axes, panels):
            if column == "longitudinal":
                values = np.asarray(
                    [float(row["accel"]) - float(row["brake"]) for row in rows],
                    dtype=float,
                )
                distances = np.asarray([float(row["dist_from_start"]) for row in rows], dtype=float)
                progress = 100.0 * (distances - distances.min()) / (distances.max() - distances.min())
                order = np.argsort(progress)
                x_values, unique_indices = np.unique(progress[order], return_index=True)
                trace = np.interp(grid, x_values, values[order][unique_indices])
            else:
                trace = profile_on_progress_grid(rows, column, grid)
            trace = smooth_profile(trace)
            ax.plot(grid, trace, color=colour, linewidth=1.65, label=label)
            ax.set_ylabel(ylabel)
            ax.grid()

    axes[1].axhline(1.0, color=GREY, linestyle=":", linewidth=1)
    axes[1].axhline(-1.0, color=GREY, linestyle=":", linewidth=1)
    axes[2].axhline(0.0, color=GREY, linewidth=0.9)
    axes[3].axhline(0.0, color=GREY, linewidth=0.9)
    axes[0].set_title("(a) Speed", loc="left", fontsize=10)
    axes[1].set_title("(b) Track position", loc="left", fontsize=10)
    axes[2].set_title("(c) Steering", loc="left", fontsize=10)
    axes[3].set_title("(d) Throttle and brake", loc="left", fontsize=10)
    axes[2].set_xlabel("Normalised lap progress (%)")
    axes[3].set_xlabel("Normalised lap progress (%)")
    axes[1].set_ylim(-1.15, 1.15)
    axes[2].set_ylim(-1.05, 1.05)
    axes[3].set_ylim(-1.05, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8.5)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(figure, "figure_10_5_control_profiles")
    plt.close(figure)


def write_tables(results: list[ResultRow]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_DIR / "chapter10_results_tables.md"
    lines = [
        "# Chapter 10 Results Tables",
        "",
        "## Table 10.1 Main Results Summary",
        "",
        "| Agent | Source | Runs | Completed | Completion rate | Best lap | Median lap | Off-track events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        best = "N/A" if row.best_lap is None else f"{row.best_lap:.3f}s"
        median = "N/A" if row.median_lap is None else f"{row.median_lap:.3f}s"
        off_track = "N/A" if row.off_track_events is None else str(row.off_track_events)
        lines.append(
            f"| {row.label} | {row.source} | {row.runs} | {row.completed} | "
            f"{row.completion_rate:.1%} | {best} | {median} | {off_track} |"
        )

    lines.extend(
        [
            "",
            "## Table 10.2 Suggested Figure List",
            "",
            "| Figure | Purpose in Chapter 10 |",
            "|---|---|",
            "| Figure 10.1 | Shows completion proportions with Wilson confidence intervals. |",
            "| Figure 10.2 | Shows completed-lap distributions with raw observations. |",
            "| Figure 10.3 | Shows pace and reliability jointly for repeated primary batches. |",
            "| Figure 10.4 | Shows material Agent 8 development milestones and their evidence. |",
            "| Figure 10.5 | Shows descriptive control profiles for representative completed laps. |",
        ]
    )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table_path)


def main() -> None:
    configure_style()
    results = main_results()
    plot_completion_with_uncertainty(results)
    plot_completed_lap_distributions()
    plot_primary_performance_map(results)
    plot_agent8_development_ladder()
    plot_representative_control_profiles()
    write_tables(results)


if __name__ == "__main__":
    main()
