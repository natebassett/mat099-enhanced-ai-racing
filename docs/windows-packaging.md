# Windows Application Package

The Windows build opens the telemetry dashboard without a terminal or manual
virtual-environment activation. TORCS remains separate from the dashboard
process and starts only when the user presses **Start**.

## Build

From the project root, install the build-only dependency once:

```powershell
.\torcs-env\Scripts\python.exe -m pip install -r packaging\windows\requirements-build.txt
```

Then build the application:

```powershell
.\scripts\build_windows_app.ps1
```

The executable is written to:

```text
dist\Enhanced AI Racing\EnhancedAIRacing.exe
```

The build script then runs a non-driving smoke test against the packaged TORCS
path, project discovery, racing line, and final Agent 7-8 policies. A failed
check stops the build instead of leaving an unverified release folder.

Distribute the complete `Enhanced AI Racing` folder. The visible executable
depends on the packaged `_internal` directory beside it.

## Included Runtime Data

The build contains TORCS, the final runtime policies for Agents 7-8, Dyna-Q
policies, racing lines, evaluation summaries, saved race results, and compact
Agent 7/8 training summaries. Replay buffers and intermediate training
checkpoints are deliberately excluded.

## Release Check

Before publishing a ZIP, test the build from a different directory and confirm:

1. The dashboard opens without a console window.
2. All seven agents appear in Agent Lab.
3. Agent 7 and Agent 8 load their packaged policies.
4. Pressing Start launches the packaged TORCS executable.
5. A completed run appears in Run History and Results.
