# Controlled agent evaluation

Valid attempts: **100/100**. Infrastructure failure records: **0**.

## Frozen comparison

All agents used TORCS `g-track-3`, car `car1-ow1`, one target lap and a 25,000-step ceiling. Attempt order was randomised within blocks and TORCS was restarted between attempts.

## Primary and secondary outcomes

| Agent | Clean laps | Rate (95% Wilson CI) | Median clean lap (IQR), s | Median distance, m |
|---|---:|---:|---:|---:|
| Rule-based | 17/20 | 85.0% (64.0-94.8%) | 98.4 (98.4-98.4) | 2944 |
| Map-aware | 17/20 | 85.0% (64.0-94.8%) | 91.0 (91.0-91.0) | 2944 |
| Dyna-Q final | 0/20 | 0.0% (0.0-16.1%) | -- | 1600 |
| Agent 7 v4 (race-line TD3) | 14/20 | 70.0% (48.1-85.5%) | 103.6 (103.4-106.4) | 2943 |
| Agent 8 final (sensor-only TD3) | 9/20 | 45.0% (25.8-65.8%) | 83.0 (83.0-83.0) | 2058 |

## Exploratory reliability comparisons

Pairwise two-sided Fisher exact tests are Holm-corrected across all pairs. They are exploratory and should be reported with effect sizes and intervals, not as the sole basis for a claim.

| Agent A | Agent B | Rate difference | Holm-adjusted p |
|---|---|---:|---:|
| Rule-based | Map-aware | +0.0 pp | 1.0000 |
| Rule-based | Dyna-Q final | +85.0 pp | 2.57e-07 |
| Rule-based | Agent 7 v4 (race-line TD3) | +15.0 pp | 1.0000 |
| Rule-based | Agent 8 final (sensor-only TD3) | +40.0 pp | 0.1122 |
| Map-aware | Dyna-Q final | +85.0 pp | 2.57e-07 |
| Map-aware | Agent 7 v4 (race-line TD3) | +15.0 pp | 1.0000 |
| Map-aware | Agent 8 final (sensor-only TD3) | +40.0 pp | 0.1122 |
| Dyna-Q final | Agent 7 v4 (race-line TD3) | -70.0 pp | 2.67e-05 |
| Dyna-Q final | Agent 8 final (sensor-only TD3) | -45.0 pp | 0.0086 |
| Agent 7 v4 (race-line TD3) | Agent 8 final (sensor-only TD3) | +25.0 pp | 0.8011 |

## Interpretation constraints

- Lap time is conditional on clean completion. A fast median from few completed laps does not establish a superior agent.
- Repetitions assess run-to-run reliability of fixed policies, not variation across independent training seeds.
- Conclusions are restricted to one track, one car and this machine/software configuration; cross-track generalisation was not tested.
- Infrastructure failures are documented separately and excluded from agent outcomes because the associated scheduled attempts remain pending until rerun.
