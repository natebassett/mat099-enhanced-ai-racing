# Documentation

This folder holds focused supporting notes for the MAT099 Enhanced AI Racing
project. The root [README](../README.md) is the main entry point for running the
application and understanding the overall project.

## Reading Path

1. Read the root README for the project aim, agent comparison, and quick start.
2. Read the Agent 7 and Agent 8 notes for the two deep reinforcement learning
   experiments and their distinct evidence boundaries.
3. Read the GUI notes when discussing explainability, results reporting, or
   accessibility in the dissertation.
4. Read the Windows packaging note when building or distributing the desktop app.

## Focused Notes

| File | Purpose |
| --- | --- |
| `agent7_n_step_td3.md` | Agent 7's racing-line-informed N-step TD3 design, experiments, and commands. |
| `agent8_sensor_n_step_td3.md` | Agent 8's sensor-only reward-driven TD3 design, reliability work, and evidence. |
| `gui_learning_visualizer.md` | Evidence boundary and research basis for the learning animation. |
| `gui_results_workspace.md` | Reporting rules for the novice-friendly Results page and detailed research evidence. |
| `gui_settings_and_sources.md` | Settings behavior, accessibility choices, and research-source boundaries. |
| `windows-packaging.md` | Reproducible build and release checks for the Windows application. |

## Supporting Assets

`assets/architecture-overview.png` is a reusable system overview. The three SVG
files are layout placeholders only; replace them with final application
screenshots before using them as dissertation figures.

## Evidence Rule

The Markdown notes explain design decisions and historical experiments. Final
performance claims should cite the saved evaluation data, telemetry, and Results
workspace rather than a single training run or a placeholder image.
