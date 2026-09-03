# Agent 7: N-Step TD3 Racer

Agent 7 is an independent continuous-control racing agent. It has separate
source files, checkpoint contracts, models, logs, and evaluation outputs from
Agent 8 and the project's engineered agents.

## Design

The implementation is informed by the experimental design and hyperparameters
published in [raphaelsenn/torcsRL](https://github.com/raphaelsenn/torcsRL), and
by the original TD3 algorithm. No source from torcsRL is copied into this
project because that repository does not currently publish a software license.
The port was reviewed against revision
`043e806e260507d1d5eef161004bc350f9d7f471`.

Agent 7 uses:

- Twin delayed deterministic policy gradients.
- Three-step discounted returns.
- Two 256-unit hidden layers in the actor and both critics.
- A replay warm-up before gradient training.
- Gaussian action exploration (`0.1` on both actions) and clipped target-policy
  smoothing, matching the torcsRL experiment. A separate longitudinal override
  remains available for controlled ablations but is not part of the default
  training protocol.
- A 45-value local telemetry/racing-line state retaining the upstream sensor
  groups plus locally derived acceleration and denser curvature look-ahead.
- Three observations and three actions of temporal history (141 values total).
- Feature-specific Gaussian observation noise matching the upstream scales:
  `0.0025` for speed, RPM, acceleration, and wheel spin; `0.00025` for angle,
  track position, track sensors, and racing-line error; no noise on look-ahead
  curvature.
- The upstream racing-line reward:
  `speed * (cos(angle) - abs(sin(angle)) - abs(line_error))`, with a fixed
  `-10` physical-failure reward.
- Automatic gear shifting outside the learned control problem.
- Full-track training from random network weights.
- Passive evaluation without actor rollback. Lap finishers are ranked by
  completion rate, median lap time, then off-track/damage tie-breakers. Distance
  is the primary ranking metric when neither policy completes a lap.

As in torcsRL, replay warm-up interactions are collected before the requested
training budget. Checkpointing and three-run deterministic median evaluation
are scheduled every 10,000 learning interactions by default. A single lucky
rollout cannot replace the protected evaluation champion.

The local TORCS SCR server does not expose torcsRL's Tita racing-line telemetry.
Agent 7 recreates those inputs from the project's racing-line JSON: target line
position, racing-line deviation, and 12 look-ahead curvature samples. Missing
acceleration channels are derived from consecutive speed samples. This makes
Agent 7 a racing-line-shaped comparison agent. It does not use teacher actions
or behaviour cloning, but it must not be described as racing-line-free.

Optional speed metadata in the local racing-line JSON is intentionally ignored;
the actor learns throttle and braking from reward.

For `g-track-3`, the app, trainer, and evaluator all prefer
`g-track-3-agent7-smooth-v1.json` and fall back to the original line only when
the smooth file is unavailable.

The default episode ceiling is 25,000 interactions, matching the reference
environment. The legacy CLI name `--observation-noise-std` represents a profile
level: `0.025` selects the reference scales above and `0` disables all
observation noise for an explicit ablation.

## Isolation

Agent 7 writes only beneath:

```text
models/agent7_n_step_td3_v3/
models/training_runs/agent7_n_step_td3_v3/
data/evaluation/agent7_n_step_td3_v3/
models/agent7_n_step_td3_v4/
models/training_runs/agent7_n_step_td3_v4/
data/evaluation/agent7_n_step_td3_v4/
models/agent7_n_step_td3_v5/
models/training_runs/agent7_n_step_td3_v5/
data/evaluation/agent7_n_step_td3_v5/
```

It does not load or overwrite Agent 8 models or earlier Agent 7 checkpoints.
The app and evaluator can still load v2 actors for comparison, but v2
actor/critic checkpoints cannot resume v3 training because their critics were
fitted to a different reward contract.

## Commands

Short v3 validation run from a fresh actor and critic:

```powershell
python scripts\train_n_step_td3_agent.py --total-timesteps 100000 --seed 0 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

Long run:

```powershell
python scripts\train_n_step_td3_agent.py --total-timesteps 1000000 --seed 0 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

Train with the optional smoother Agent 7 racing line while retaining the
original `g-track-3.json`:

```powershell
python scripts\train_n_step_td3_agent.py --total-timesteps 1000000 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json
```

The improved line can be regenerated reproducibly with:

```powershell
python scripts\generate_agent7_smooth_racing_line.py
```

Continue a healthy v3 evaluation champion with a fresh replay buffer. Evaluation
only saves checkpoints; it does not alter the live learner:

```powershell
python scripts\train_n_step_td3_agent.py --resume-best --total-timesteps 100000 --seed 1 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

For short pace refinement after a reliable lap champion exists, use the explicit
fine-tuning mode. It preserves the loaded actor, critic, target networks, and
optimizer moments while reducing the actor learning rate to `3e-6` by default, both
action-noise scales to `0.01`, updating the actor every fourth critic update,
and running one gradient batch every eight simulator interactions by default. Pace
continuations first collect 30,000 transitions with deterministic actions from
the fixed actor, perform 2,000 critic-only updates, and only then enable `0.01`
action exploration and resume low-rate actor learning:

```powershell
python scripts\train_n_step_td3_agent.py --resume-best --fine-tune --fine-tune-actor-learning-rate 3e-6 --total-timesteps 2500 --train-frequency 32 --seed 15 --no-tensorboard
```

With 2,500 learning interactions, train frequency 32, and policy delay 4, this
ultra-conservative profile permits roughly 20 actor updates before the final
passive evaluation. Use it only for refinement of a mature validated policy;
the learning-rate option must be positive and finite.

### Steering-rate v4 experiment

The v4 experiment addresses the observed full-scale left/right switching with
one deliberately small reward change:

```text
reward_v4 = reward_v3 - 0.0025 * (steer_t - steer_t-1)^2
```

Forward-velocity and racing-line terms are unchanged. There is no steering
filter, straight detector, speed target, teacher action, actor rollback, or
hardcoded driving intervention. The actor still controls the car directly.

V4 starts from the protected v3 `best_evaluation.pt` actor parameters only. It
creates a fresh critic, target critic, optimizers, replay buffer, counters, and
source-evaluation state. The default run collects 30,000 deterministic
transitions, adapts the critic for 5,000 updates with the actor frozen, then
allows at most 10 actor updates at learning rate `1e-6`. Critic learning can
continue after the actor ceiling. A candidate is promoted only after 10/10
internal lap completions with median lap time strictly below `103.953 s`.

Run one bounded v4 seed with:

```powershell
python scripts\train_n_step_td3_agent.py --resume-best --steering-rate-v4 --seed 19 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

V4 artifacts are isolated under the v4 paths above. Existing v3 checkpoints
are never overwritten, and an existing v4 champion is retained unless a later
seed passes the same gate and improves its evaluation score.

Evaluate a promoted v4 candidate independently with:

```powershell
python scripts\evaluate_n_step_td3_agent.py --policy-path models\agent7_n_step_td3_v4\best_evaluation.pt --repeats 10 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json
```

### Pace v5 experiment

V5 is a separate fresh-training contract intended to challenge the map-aware
agent without changing the reliable v3 policy. It retains the same TD3 network,
three-step replay, full-track episodes, telemetry history, and racing-line
curvature observations. It makes only two control/objective changes:

```text
physical_steer = clip(actor_steer, -1, 1) * 0.3
reward_v5 = clipped_signed_progress_m - 10 * physical_failure
            + 1000 * lap_completed
```

The progress delta is clipped to `+/-5 m` per interaction to reject impossible
telemetry jumps. V5 has no racing-line adherence reward, target-speed feature,
teacher action, behaviour cloning, curriculum, actor rollback, safety probe, or
hardcoded control intervention. The racing-line JSON speed fields are ignored;
only line geometry is observed.

V5's initial random replay collection holds each sampled action for 10
interactions (about 200 ms at 50 Hz). Eighty percent of those segments apply
positive longitudinal input and 20 percent explore braking or coasting. This
keeps the warm-up reward-free and stochastic while ensuring its replay contains
launches instead of 50 Hz throttle/brake alternation.

V5 must start with fresh actor/critic weights and fresh replay because the
bounded physical action and progress objective are new contracts. The trainer
rejects resume checkpoints and replay imports in this mode. Its default budget
is one million learning interactions, with passive three-run evaluation every
50,000 interactions. Evaluation may save a better reliability or pace
checkpoint but never changes the live learner.

Run a short contract validation before committing to an overnight run:

```powershell
python scripts\train_n_step_td3_agent.py --pace-v5 --total-timesteps 50000 --seed 0 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

Run the intended one-million-interaction experiment:

```powershell
python scripts\train_n_step_td3_agent.py --pace-v5 --seed 0 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

Evaluate the protected v5 reliability champion independently:

```powershell
python scripts\evaluate_n_step_td3_agent.py --policy-path models\agent7_n_step_td3_v5\best_evaluation.pt --repeats 10 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json
```

Evaluation keeps two independent full checkpoints. `best_evaluation.pt` is the
reliability champion selected by completion rate and then median lap time.
`best_pace.pt` is the fastest clean deterministic evaluation lap, even when that
candidate is less reliable. Pace selection is passive and never replaces the
live learner or the reliability champion. Continue pace exploration explicitly:

```powershell
python scripts\train_n_step_td3_agent.py --resume-pace --fine-tune --total-timesteps 2500 --seed 6 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json --no-tensorboard
```

Do not provide `--replay-buffer-path` for collapse recovery. That option is
retained only for exact continuation of a healthy interrupted run. The default
`--train-frequency 4` is a conservative local adaptation for the highly
correlated 50 Hz Windows TORCS stream; use `--train-frequency 1` only when
deliberately reproducing the upstream update cadence.

Save and later restore replay memory when disk space permits:

```powershell
python scripts\train_n_step_td3_agent.py --total-timesteps 500000 --save-replay-buffer
python scripts\train_n_step_td3_agent.py --resume models\agent7_n_step_td3_v3\latest.pt --replay-buffer-path models\agent7_n_step_td3_v3\latest.replay.pt --total-timesteps 500000
```

Deterministic three-run evaluation:

```powershell
python scripts\evaluate_n_step_td3_agent.py --policy-path models\agent7_n_step_td3_v3\best_evaluation.pt --repeats 3 --racing-line-path data\racing_lines\g-track-3-agent7-smooth-v1.json
```
