# Chapter 10 Results Tables

## Table 10.1 Main Results Summary

| Agent | Source | Runs | Completed | Completion rate | Best lap | Median lap | Off-track events |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based anti-spin | application DB | 2 | 2 | 100.0% | 98.394s | 99.142s | 0 |
| Map-aware racing line | application DB | 15 | 15 | 100.0% | 90.958s | 92.056s | 0 |
| Dyna-Q learning | application DB | 3 | 1 | 33.3% | 305.562s | 305.562s | 923 |
| TD3 scratch | application DB | 6 | 0 | 0.0% | N/A | N/A | 216 |
| Agent 7 TD3 v3 | 20-run evaluation | 20 | 18 | 90.0% | 103.230s | 104.220s | 2 |
| Agent 7 TD3 v4 | 10-run evaluation | 10 | 10 | 100.0% | 103.386s | 103.887s | 0 |
| Agent 8 sensor-only TD3 | two 20-run evaluations | 40 | 36 | 90.0% | 83.038s | 83.038s | 4 |

## Table 10.2 Suggested Figure List

| Figure | Purpose in Chapter 10 |
|---|---|
| Figure 10.1 | Shows completion proportions with Wilson confidence intervals. |
| Figure 10.2 | Shows completed-lap distributions with raw observations. |
| Figure 10.3 | Shows pace and reliability jointly for repeated primary batches. |
| Figure 10.4 | Shows material Agent 8 development milestones and their evidence. |
| Figure 10.5 | Shows descriptive control profiles for representative completed laps. |
