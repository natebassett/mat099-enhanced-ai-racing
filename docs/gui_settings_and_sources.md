# Settings, Research Evidence, and Sources

The Settings page keeps occasional choices and dissertation-level evidence away
from the application's everyday race controls. It does not modify agents,
checkpoints, training logs, evaluation files, or TORCS configuration.

## Preferences

- **Race defaults** select the driver and track shown on the next application
  launch. Choosing "Keep the current driver" preserves the normal discovery
  order.
- **Opening screen** selects Live Telemetry, Results, or Agent Lab as the first
  page shown on the next launch.
- **Live chart history** changes how many seconds remain visible in the live
  telemetry charts. This takes effect as soon as settings are saved.
- **Reduce animated movement** replaces automatic playback in the learning
  visualiser with one-step-at-a-time progression. This takes effect immediately.
- **Theme** selects the Windows theme, a fixed light theme, or a fixed dark
  theme. The application shell and telemetry plots update together when the
  setting is saved.
- **Colour presentation** provides standard, colour-accessible, and
  high-contrast palettes. Plot series also use different line styles and marker
  shapes so colour is not the only way to distinguish results.
- **Language** switches the novice-facing application controls between English
  and Welsh. Agent names, model identifiers, measurements, and source titles
  remain unchanged so recorded evidence stays unambiguous.
- **Helpful notices** controls whether the measured TD3 reliability note appears
  before a learned driver starts. Choosing "Do not show this note again" in the
  notice changes the same preference and never changes the policy or race setup.

Preferences are stored in `data/generated/gui_settings.json`. Missing, malformed,
or out-of-range values fall back to conservative defaults. Saving uses a temporary
file followed by replacement so an interrupted write cannot leave a partial JSON
document.

## Data and Storage

The storage controls intentionally separate disposable application state from
experiment evidence:

- **Clear temporary cache** removes Python `__pycache__` directories under
  `src/` and `scripts/`, then clears the learning visualiser's in-memory
  checkpoint summary cache. It does not scan or alter `data/`, `models/`, the
  Python environment, or the TORCS wrapper.
- **Reset run history** deletes older GUI race records from SQLite in one
  transaction. It keeps one representative replay for every agent type,
  preferring a completed lap on the selected default track, then the fastest
  completed lap, then the longest incomplete run.
- Models, replay buffers, evaluation JSON/CSV files, training logs, recorded
  laps, racing lines, and source code are always protected from both controls.

Run-history reset is deliberately confirmed with a destructive-action warning.
Dependent GUI telemetry and metrics belonging to deleted runs are removed by
SQLite foreign-key cascades; a database error rolls the entire operation back.

## TD3 Reliability Notice

Agent 8's pre-race notice reports the project's observed `19/20` repeated-run
completion result. It is an evidence label for that evaluated checkpoint, not a
claim that all TD3 policies fail exactly one race in twenty. Agent 7 has its own
policy and evaluation contract, so its reliability is reported separately.

## Research Evidence

The Research Evidence tab contains the detailed evaluation workspace removed from
the novice-facing Results page. It preserves batch selection, individual trial
outcomes, policy provenance, and observation/action/reward contract identifiers.
This is the appropriate view for dissertation analysis; the main Results page is
the appropriate view for demonstrations and first-time users.

## Sources and Methods

The source library links each interface or algorithm decision to its research
basis. The cited material covers:

- TD3's twin critics, delayed actor updates, and target policy smoothing;
- the TORCS reinforcement-learning implementation reviewed for Agent 7;
- repeated-trial and reproducibility concerns in deep RL evaluation; and
- progressive disclosure in educational neural-network visualisation.
- WCAG guidance on contrast and avoiding colour-only communication; and
- Qt's runtime translation framework and established localisation model.

The source library is explanatory only. Opening a source or local methodology
guide never changes application or experiment state.
