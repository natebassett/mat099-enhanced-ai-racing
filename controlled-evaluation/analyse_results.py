#!/usr/bin/env python3
"""Analyse a completed controlled-evaluation session and create paper figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_ROOT = Path(__file__).with_name("results")
SUMMARY_FIELDS = [
    "agent_id",
    "agent_label",
    "attempts",
    "clean_completions",
    "completion_rate",
    "completion_ci_low",
    "completion_ci_high",
    "clean_lap_median_s",
    "clean_lap_q1_s",
    "clean_lap_q3_s",
    "clean_lap_bootstrap_ci_low_s",
    "clean_lap_bootstrap_ci_high_s",
    "distance_median_m",
    "failed_distance_median_m",
    "mean_speed_median_kmh",
    "damage_delta_median",
]

PAIRWISE_FIELDS = [
    "agent_a",
    "agent_b",
    "completion_rate_difference",
    "odds_ratio",
    "fisher_exact_p",
    "holm_adjusted_p",
    "interpretation",
]

EVENT_FIELDS = [
    "agent_id",
    "agent_label",
    "attempts",
    "attempts_with_off_track",
    "attempts_with_reverse",
    "attempts_with_stall",
    "attempts_with_unsafe_heading",
    "attempts_with_damage_increase",
    "completed_unclean",
]

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
FAILURE_COLOURS = {
    "Clean completion": "#009E73",
    "Completed unclean": "#E69F00",
    "Off track": "#D55E00",
    "Reverse": "#CC79A7",
    "Stall": "#0072B2",
    "Unsafe heading": "#56B4E9",
    "Crash": "#000000",
    "Max steps": "#8A8A8A",
    "Other": "#BDBDBD",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = [finite_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def wilson_interval(successes: int, attempts: int, z: float = 1.9599639845) -> tuple[float, float]:
    if attempts <= 0:
        return math.nan, math.nan
    proportion = successes / attempts
    denominator = 1.0 + z * z / attempts
    centre = (proportion + z * z / (2.0 * attempts)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / attempts
            + z * z / (4.0 * attempts * attempts)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def bootstrap_median_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        value = float(array[0])
        return value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(resamples, len(array)))
    medians = np.median(array[indices], axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def summarise_agents(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    design = protocol["design"]
    resamples = int(design.get("bootstrap_resamples", 10000))
    base_seed = int(design.get("order_seed", 0)) + 1701
    summaries = []
    for index, agent in enumerate(protocol["agents"]):
        agent_rows = [row for row in rows if row.get("agent_id") == agent["id"]]
        clean_rows = [row for row in agent_rows if csv_bool(row.get("clean_completion"))]
        failed_rows = [row for row in agent_rows if not csv_bool(row.get("clean_completion"))]
        clean_laps = numeric_values(clean_rows, "lap_time_seconds")
        successes = len(clean_rows)
        attempts = len(agent_rows)
        ci_low, ci_high = wilson_interval(successes, attempts)
        boot_low, boot_high = bootstrap_median_interval(
            clean_laps,
            resamples=resamples,
            seed=base_seed + index,
        )
        summaries.append(
            {
                "agent_id": agent["id"],
                "agent_label": agent["label"],
                "attempts": attempts,
                "clean_completions": successes,
                "completion_rate": successes / attempts if attempts else None,
                "completion_ci_low": ci_low if attempts else None,
                "completion_ci_high": ci_high if attempts else None,
                "clean_lap_median_s": _median(clean_laps),
                "clean_lap_q1_s": _quantile(clean_laps, 0.25),
                "clean_lap_q3_s": _quantile(clean_laps, 0.75),
                "clean_lap_bootstrap_ci_low_s": boot_low,
                "clean_lap_bootstrap_ci_high_s": boot_high,
                "distance_median_m": _median(numeric_values(agent_rows, "distance_m")),
                "failed_distance_median_m": _median(
                    numeric_values(failed_rows, "distance_m")
                ),
                "mean_speed_median_kmh": _median(
                    numeric_values(agent_rows, "mean_speed_kmh")
                ),
                "damage_delta_median": _median(
                    numeric_values(agent_rows, "damage_delta")
                ),
            }
        )
    return summaries


def event_incidence(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for agent in protocol["agents"]:
        agent_rows = [row for row in rows if row.get("agent_id") == agent["id"]]
        output.append(
            {
                "agent_id": agent["id"],
                "agent_label": agent["label"],
                "attempts": len(agent_rows),
                "attempts_with_off_track": _count_positive(agent_rows, "off_track_steps"),
                "attempts_with_reverse": _count_positive(agent_rows, "reverse_steps"),
                "attempts_with_stall": sum(
                    _display_failure_category(row) == "Stall" for row in agent_rows
                ),
                "attempts_with_unsafe_heading": sum(
                    _display_failure_category(row) == "Unsafe heading"
                    for row in agent_rows
                ),
                "attempts_with_damage_increase": _count_positive(
                    agent_rows, "damage_delta"
                ),
                "completed_unclean": sum(
                    row.get("failure_category") == "completed_unclean"
                    for row in agent_rows
                ),
            }
        )
    return output


def _count_positive(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum((finite_float(row.get(key)) or 0.0) > 0.0 for row in rows)


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    return float(np.quantile(values, probability)) if values else None


def fisher_exact_two_sided(
    a: int, b: int, c: int, d: int
) -> tuple[float | str | None, float]:
    row_one = a + b
    row_two = c + d
    successes = a + c
    total = row_one + row_two
    denominator = math.comb(total, successes)

    def probability(x: int) -> float:
        return math.comb(row_one, x) * math.comb(row_two, successes - x) / denominator

    minimum = max(0, successes - row_two)
    maximum = min(row_one, successes)
    observed = probability(a)
    p_value = sum(
        probability(x)
        for x in range(minimum, maximum + 1)
        if probability(x) <= observed + 1e-15
    )
    if b * c == 0:
        if a * d == 0:
            odds_ratio: float | str | None = None
        else:
            odds_ratio = "inf"
    else:
        odds_ratio = (a * d) / (b * c)
    return odds_ratio, min(1.0, p_value)


def pairwise_reliability(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw = []
    for agent_a, agent_b in combinations(protocol["agents"], 2):
        rows_a = [row for row in rows if row.get("agent_id") == agent_a["id"]]
        rows_b = [row for row in rows if row.get("agent_id") == agent_b["id"]]
        success_a = sum(csv_bool(row.get("clean_completion")) for row in rows_a)
        success_b = sum(csv_bool(row.get("clean_completion")) for row in rows_b)
        if not rows_a or not rows_b:
            continue
        odds_ratio, p_value = fisher_exact_two_sided(
            success_a,
            len(rows_a) - success_a,
            success_b,
            len(rows_b) - success_b,
        )
        raw.append(
            {
                "agent_a": agent_a["label"],
                "agent_b": agent_b["label"],
                "completion_rate_difference": success_a / len(rows_a)
                - success_b / len(rows_b),
                "odds_ratio": odds_ratio,
                "fisher_exact_p": p_value,
            }
        )

    ordered = sorted(enumerate(raw), key=lambda item: item[1]["fisher_exact_p"])
    running_max = 0.0
    adjusted: dict[int, float] = {}
    comparison_count = len(raw)
    for rank, (original_index, item) in enumerate(ordered):
        corrected = min(1.0, (comparison_count - rank) * item["fisher_exact_p"])
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    for index, item in enumerate(raw):
        item["holm_adjusted_p"] = adjusted[index]
        item["interpretation"] = (
            "exploratory difference after Holm correction"
            if adjusted[index] < 0.05
            else "no exploratory difference after Holm correction"
        )
    return raw


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_completion_intervals(
    summaries: Sequence[Mapping[str, Any]], figures_dir: Path
) -> None:
    available = [row for row in summaries if int(row["attempts"]) > 0]
    labels = [str(row["agent_label"]) for row in available]
    rates = np.asarray([float(row["completion_rate"]) * 100 for row in available])
    lows = np.asarray([float(row["completion_ci_low"]) * 100 for row in available])
    highs = np.asarray([float(row["completion_ci_high"]) * 100 for row in available])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for index, (position, rate, low, high) in enumerate(zip(y, rates, lows, highs)):
        ax.errorbar(
            rate,
            position,
            xerr=[[rate - low], [high - rate]],
            fmt="o",
            markersize=6,
            color=PALETTE[index % len(PALETTE)],
            ecolor=PALETTE[index % len(PALETTE)],
            capsize=4,
            linewidth=1.8,
        )
        row = available[index]
        ax.text(
            min(101.5, high + 2.0),
            position,
            f"{row['clean_completions']}/{row['attempts']}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(-2, 108)
    ax.set_xlabel("Clean completion rate (%) with 95% Wilson interval")
    ax.set_title("Clean-lap reliability under the fixed evaluation protocol")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    save_figure(fig, figures_dir, "completion_reliability_intervals")


def plot_lap_distributions(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> None:
    labels = []
    data = []
    for agent in protocol["agents"]:
        laps = numeric_values(
            [
                row
                for row in rows
                if row.get("agent_id") == agent["id"]
                and csv_bool(row.get("clean_completion"))
            ],
            "lap_time_seconds",
        )
        if laps:
            labels.append(f"{plot_agent_label(agent['label'])}\n(n={len(laps)})")
            data.append(laps)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    if data:
        box = ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
            widths=0.55,
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.6},
        )
        generator = np.random.default_rng(20260906)
        for index, (patch, values) in enumerate(zip(box["boxes"], data), start=1):
            colour = PALETTE[(index - 1) % len(PALETTE)]
            patch.set_facecolor(colour)
            patch.set_alpha(0.28)
            patch.set_edgecolor(colour)
            jitter = generator.normal(0.0, 0.045, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=24,
                alpha=0.82,
                color=colour,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        ax.set_ylabel("Lap time among clean completions (s)")
    else:
        ax.text(0.5, 0.5, "No clean lap times recorded", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Pace distribution, conditional on clean completion")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, figures_dir, "clean_lap_time_distribution")


def plot_agent_label(label: str) -> str:
    return str(label).replace(" (race-line TD3)", "\n(race-line TD3)").replace(
        " (sensor-only TD3)", "\n(sensor-only TD3)"
    )


def _display_failure_category(row: Mapping[str, Any]) -> str:
    category = str(row.get("failure_category") or "other")
    mapping = {
        "clean_completion": "Clean completion",
        "completed_unclean": "Completed unclean",
        "off_track": "Off track",
        "reverse": "Reverse",
        "stall": "Stall",
        "stuck": "Stall",
        "car stalled": "Stall",
        "unsafe heading": "Unsafe heading",
        "crash": "Crash",
        "crashed": "Crash",
        "max_steps": "Max steps",
    }
    return mapping.get(category, "Other")


def plot_failure_composition(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> None:
    categories = list(FAILURE_COLOURS)
    labels = [agent["label"] for agent in protocol["agents"]]
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for category in categories:
        percentages = []
        counts = []
        for agent in protocol["agents"]:
            agent_rows = [row for row in rows if row.get("agent_id") == agent["id"]]
            count = sum(_display_failure_category(row) == category for row in agent_rows)
            counts.append(count)
            percentages.append(100.0 * count / len(agent_rows) if agent_rows else 0.0)
        if not any(counts):
            continue
        bars = ax.barh(
            y,
            percentages,
            left=left,
            height=0.58,
            label=category,
            color=FAILURE_COLOURS[category],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, percentage, count in zip(bars, percentages, counts):
            if percentage >= 9:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if category in {"Off track", "Reverse", "Stall", "Crash"} else "#111111",
                )
        left += np.asarray(percentages)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of valid attempts (%)")
    ax.set_title("Outcome composition and primary failure classification")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, figures_dir, "outcome_composition")


def plot_distance_ecdf(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for index, agent in enumerate(protocol["agents"]):
        values = np.sort(
            numeric_values(
                [row for row in rows if row.get("agent_id") == agent["id"]],
                "distance_m",
            )
        )
        if len(values) == 0:
            continue
        probabilities = np.arange(1, len(values) + 1) / len(values)
        ax.step(
            values,
            probabilities,
            where="post",
            linewidth=1.8,
            color=PALETTE[index % len(PALETTE)],
            label=agent["label"],
        )
        ax.scatter(
            values,
            probabilities,
            s=12,
            color=PALETTE[index % len(PALETTE)],
        )
    ax.set_xlabel("Distance raced (m)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0, 1.03)
    ax.set_title("Distribution of progress across all valid attempts")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, figures_dir, "distance_raced_ecdf")


def plot_reliability_pace(
    summaries: Sequence[Mapping[str, Any]], figures_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    plotted = 0
    available_laps = [
        lap
        for row in summaries
        if (lap := finite_float(row.get("clean_lap_median_s"))) is not None
    ]
    fastest_lap = min(available_laps) if available_laps else None
    for index, row in enumerate(summaries):
        lap = finite_float(row.get("clean_lap_median_s"))
        if lap is None:
            continue
        rate = float(row["completion_rate"]) * 100
        low = float(row["completion_ci_low"]) * 100
        high = float(row["completion_ci_high"]) * 100
        lap_low = finite_float(row.get("clean_lap_bootstrap_ci_low_s")) or lap
        lap_high = finite_float(row.get("clean_lap_bootstrap_ci_high_s")) or lap
        ax.errorbar(
            rate,
            lap,
            xerr=[[rate - low], [high - rate]],
            yerr=[[lap - lap_low], [lap_high - lap]],
            fmt="o",
            markersize=7,
            capsize=3,
            linewidth=1.3,
            color=PALETTE[index % len(PALETTE)],
        )
        ax.annotate(
            str(row["agent_label"]),
            (rate, lap),
            xytext=(6, -15 if lap == fastest_lap else 6),
            textcoords="offset points",
            fontsize=8,
        )
        plotted += 1
    if plotted:
        ax.set_xlim(-2, 108)
        ax.invert_yaxis()
        ax.set_xlabel("Clean completion rate (%)")
        ax.set_ylabel("Median clean lap time (s; faster is higher)")
    else:
        ax.text(0.5, 0.5, "No reliability-pace points available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Reliability-pace trade-off with uncertainty")
    ax.margins(y=0.12)
    save_figure(fig, figures_dir, "reliability_pace_tradeoff")


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def format_number(value: Any, digits: int = 1, missing: str = "--") -> str:
    number = finite_float(value)
    return missing if number is None else f"{number:.{digits}f}"


def format_p_value(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "--"
    if 0.0 < number < 0.0001:
        return f"{number:.2e}"
    return f"{number:.4f}"


def write_latex_table(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Agent & Clean laps & Rate (95\% CI) & Median lap (s) & IQR (s) \\",
        r"\midrule",
    ]
    for row in summaries:
        if int(row["attempts"]) > 0:
            rate_text = (
                f"{float(row['completion_rate']) * 100:.1f} "
                f"({float(row['completion_ci_low']) * 100:.1f}--"
                f"{float(row['completion_ci_high']) * 100:.1f})"
            )
        else:
            rate_text = "--"
        iqr = (
            f"{format_number(row['clean_lap_q1_s'])}--{format_number(row['clean_lap_q3_s'])}"
            if row["clean_lap_q1_s"] is not None
            else "--"
        )
        lines.append(
            f"{latex_escape(row['agent_label'])} & "
            f"{row['clean_completions']}/{row['attempts']} & "
            f"{rate_text} & "
            f"{format_number(row['clean_lap_median_s'])} & {iqr} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(
    path: Path,
    protocol: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    *,
    attempts: int,
    scheduled: int,
    infrastructure_failures: int,
) -> None:
    environment = protocol["environment"]
    lines = [
        "# Controlled agent evaluation",
        "",
        f"Valid attempts: **{attempts}/{scheduled}**. Infrastructure failure records: "
        f"**{infrastructure_failures}**.",
        "",
        "## Frozen comparison",
        "",
        f"All agents used TORCS `{environment['track_id']}`, car "
        f"`{environment['car_id']}`, one target lap and a "
        f"{int(environment['max_steps']):,}-step ceiling. Attempt order was randomised "
        "within blocks and TORCS was restarted between attempts.",
        "",
        "## Primary and secondary outcomes",
        "",
        "| Agent | Clean laps | Rate (95% Wilson CI) | Median clean lap (IQR), s | Median distance, m |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if int(row["attempts"]) > 0:
            rate_text = (
                f"{float(row['completion_rate']) * 100:.1f}% "
                f"({float(row['completion_ci_low']) * 100:.1f}-"
                f"{float(row['completion_ci_high']) * 100:.1f}%)"
            )
        else:
            rate_text = "--"
        if row["clean_lap_median_s"] is None:
            lap_summary = "--"
        else:
            lap_summary = (
                f"{format_number(row['clean_lap_median_s'])} "
                f"({format_number(row['clean_lap_q1_s'])}-"
                f"{format_number(row['clean_lap_q3_s'])})"
            )
        lines.append(
            f"| {row['agent_label']} | {row['clean_completions']}/{row['attempts']} | "
            f"{rate_text} | {lap_summary} | "
            f"{format_number(row['distance_median_m'], 0)} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory reliability comparisons",
            "",
            "Pairwise two-sided Fisher exact tests are Holm-corrected across all pairs. "
            "They are exploratory and should be reported with effect sizes and intervals, "
            "not as the sole basis for a claim.",
            "",
            "| Agent A | Agent B | Rate difference | Holm-adjusted p |",
            "|---|---|---:|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            f"| {row['agent_a']} | {row['agent_b']} | "
            f"{float(row['completion_rate_difference']) * 100:+.1f} pp | "
            f"{format_p_value(row['holm_adjusted_p'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Lap time is conditional on clean completion. A fast median from few completed "
            "laps does not establish a superior agent.",
            "- Repetitions assess run-to-run reliability of fixed policies, not variation "
            "across independent training seeds.",
            "- Conclusions are restricted to one track, one car and this machine/software "
            "configuration; cross-track generalisation was not tested.",
            "- Infrastructure failures are documented separately and excluded from agent "
            "outcomes because the associated scheduled attempts remain pending until rerun.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_session(
    session_dir: Path, *, allow_incomplete: bool
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], int]:
    required = [
        session_dir / "protocol_snapshot.json",
        session_dir / "schedule.csv",
        session_dir / "raw_attempts.csv",
        session_dir / "infrastructure_failures.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Session files are missing: " + ", ".join(missing))
    protocol = load_json(session_dir / "protocol_snapshot.json")
    schedule = read_csv_rows(session_dir / "schedule.csv")
    rows = read_csv_rows(session_dir / "raw_attempts.csv")
    infrastructure_count = len(
        read_csv_rows(session_dir / "infrastructure_failures.csv")
    )
    scheduled_ids = {row["attempt_id"] for row in schedule}
    attempt_ids = [row["attempt_id"] for row in rows]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("raw_attempts.csv contains duplicate attempt ids")
    unexpected = sorted(set(attempt_ids) - scheduled_ids)
    if unexpected:
        raise ValueError(f"Unexpected attempt ids: {', '.join(unexpected)}")
    if len(rows) != len(schedule) and not allow_incomplete:
        raise ValueError(
            f"Session is incomplete ({len(rows)}/{len(schedule)} valid attempts). "
            "Resume it or pass --allow-incomplete for diagnostic analysis."
        )
    if not rows:
        raise ValueError("No valid attempts are available for analysis")
    return protocol, schedule, rows, infrastructure_count


def analyse_session(session_dir: Path, *, allow_incomplete: bool = False) -> Path:
    protocol, schedule, rows, infrastructure_count = validate_session(
        session_dir, allow_incomplete=allow_incomplete
    )
    analysis_dir = session_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()
    summaries = summarise_agents(protocol, rows)
    events = event_incidence(protocol, rows)
    pairwise = pairwise_reliability(protocol, rows)

    write_csv(analysis_dir / "summary_statistics.csv", SUMMARY_FIELDS, summaries)
    write_csv(analysis_dir / "event_incidence.csv", EVENT_FIELDS, events)
    write_csv(analysis_dir / "pairwise_reliability.csv", PAIRWISE_FIELDS, pairwise)
    write_latex_table(analysis_dir / "summary_table.tex", summaries)
    write_markdown_report(
        analysis_dir / "analysis.md",
        protocol,
        summaries,
        pairwise,
        attempts=len(rows),
        scheduled=len(schedule),
        infrastructure_failures=infrastructure_count,
    )
    (analysis_dir / "analysis.json").write_text(
        json.dumps(
            {
                "session_id": protocol.get("session", {}).get("session_id"),
                "complete": len(rows) == len(schedule),
                "valid_attempts": len(rows),
                "scheduled_attempts": len(schedule),
                "infrastructure_failure_records": infrastructure_count,
                "summary": summaries,
                "event_incidence": events,
                "pairwise_reliability": pairwise,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_completion_intervals(summaries, figures_dir)
    plot_lap_distributions(protocol, rows, figures_dir)
    plot_failure_composition(protocol, rows, figures_dir)
    plot_distance_ecdf(protocol, rows, figures_dir)
    plot_reliability_pace(summaries, figures_dir)
    return analysis_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create statistical tables and figures for an evaluation session."
    )
    parser.add_argument("--session", required=True, help="Prepared session id.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate diagnostic outputs before all attempts are complete.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.session):
            raise ValueError("Invalid session id")
        session_dir = args.results_root.resolve() / args.session
        output = analyse_session(session_dir, allow_incomplete=args.allow_incomplete)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Analysis written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
