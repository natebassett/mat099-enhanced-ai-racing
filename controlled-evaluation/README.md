# Controlled agent evaluation

This folder evaluates the final driving agents under one frozen protocol without
changing the GUI, agent implementations, TORCS runner or packaged executable.
The intended dissertation claim is comparative and deliberately narrow: results
apply to one car (`car1-ow1`), one track (`g-track-3`) and one-lap attempts.

## Protocol

`protocol_v1.json` pre-declares the agents, checkpoints, environment, outcomes,
sample size and analysis. Each of 20 blocks contains one attempt by every agent;
order inside each block is deterministically randomised. TORCS is restarted for
every attempt. A clean completion requires a timed lap with no off-track sample,
reverse sample or stall termination.

The runner records a manifest, hashes the protocol and all model/data assets,
stores a deterministic schedule, writes one compressed telemetry file per valid
attempt, and updates `raw_attempts.csv` atomically. Simulator or connection
errors go to `infrastructure_failures.csv`; they remain pending and are retried
when the same session is resumed. They are never counted as agent failures.

## Before the final experiment

1. Close the GUI and any existing TORCS process.
2. Commit this evaluation tooling so the manifest can identify a clean revision.
3. Keep the machine plugged in and avoid other heavy applications.
4. Do not edit `protocol_v1.json`, model files or racing-line files after the
   session has been prepared.

## Commands

From the repository root, first run a short four-attempt pipeline check:

```powershell
.\torcs-env\Scripts\python.exe controlled-evaluation\run_evaluation.py smoke --session-id pipeline-smoke
.\torcs-env\Scripts\python.exe controlled-evaluation\analyse_results.py --session pipeline-smoke
```

The smoke command runs two attempts each for Agent 7 and Agent 8. It is a tooling
check only and must not be reported as the final experiment.

Prepare and run the complete pre-declared experiment:

```powershell
.\torcs-env\Scripts\python.exe controlled-evaluation\run_evaluation.py prepare --session-id final-agent-evaluation
.\torcs-env\Scripts\python.exe controlled-evaluation\run_evaluation.py run --session final-agent-evaluation
.\torcs-env\Scripts\python.exe controlled-evaluation\analyse_results.py --session final-agent-evaluation
```

An interrupted final run resumes with the same `run` command. Check progress at
any time without starting TORCS:

```powershell
.\torcs-env\Scripts\python.exe controlled-evaluation\run_evaluation.py status --session final-agent-evaluation
```

Outputs are written beneath `controlled-evaluation/results/<session-id>/` and
are ignored by Git by default. Review the generated `analysis/analysis.md`, CSV
tables, LaTeX table and PNG/PDF figures before deliberately adding a final
evidence session to version control.

## Interpretation boundary

Repeated runs estimate reliability under the fixed local setup; they are not
independent retraining experiments. Lap-time summaries include clean completions
only, so they must always be presented beside completion rate. This protocol
does not establish generalisation to other tracks, cars, computers or users.
