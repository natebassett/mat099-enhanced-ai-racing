# Agent 8: Sensor-Only N-Step TD3 Racer

Agent 8 is a separate reward-only experiment derived from the stable Agent 7
N-step TD3 implementation. It shares the same actor, twin critics, three-step
replay, target-policy smoothing, delayed actor updates, full-track episodes,
checkpointing, CSV telemetry, and passive deterministic evaluation.

It does not load or inspect a racing-line JSON file. The policy receives only:

- TORCS speed and derived acceleration;
- RPM, heading angle, and normalized track position;
- all 19 TORCS track-edge range sensors;
- four wheel-speed sensors;
- three observations and three previous actions of history.

The existing 141-input network shape is retained so this is a controlled
comparison with Agent 7. The 13 slots previously used by racing-line deviation
and curvature preview are always zero and carry no information.

The reward is the original v3 velocity objective with the target fixed to the
track centre rather than a saved line:

```text
reward = (speed_x / 250) * (cos(angle) - abs(sin(angle)) - abs(track_pos))
failure = -10
```

Automatic gear shifting remains outside the learned action. The actor learns
normalized steering and one signed longitudinal command: positive values mean
throttle and negative values mean braking. There is no teacher, behaviour
cloning, imitation, target speed, curriculum, actor rollback, or control
intervention.

Fresh runs fill the initial replay buffer with held random action segments. A
segment lasts 10 interactions, uses an 80% throttle and 20% brake/coast split,
and limits random warm-up steering to +/-0.3. This training-only exploration
profile gives the replay buffer useful moving transitions without prescribing
how the learned policy drives. The actor retains the full normalized steering
contract during training, evaluation, and application races.

Agent 8 writes only beneath:

```text
models/agent8_sensor_n_step_td3/
models/training_runs/agent8_sensor_n_step_td3/
data/evaluation/agent8_sensor_n_step_td3/
```

Run a short fresh validation:

```powershell
python scripts\train_sensor_n_step_td3_agent.py --total-timesteps 50000 --seed 0 --no-tensorboard
```

Run a longer experiment after validating the logs:

```powershell
python scripts\train_sensor_n_step_td3_agent.py --total-timesteps 1000000 --seed 0 --no-tensorboard
```

Evaluate the protected deterministic champion:

```powershell
python scripts\evaluate_sensor_n_step_td3_agent.py --repeats 10
```

The console application exposes the trained policy as option `8`. Agent 7 and
Agent 8 checkpoints have distinct model, observation, and reward contracts, so
they cannot be loaded into one another accidentally.

## V1 continuation

The protected `champion_100_942s_internal.pt` pace frontier remains immutable.
The V1 continuation defaults to that full checkpoint and resumes its complete
TD3 state: actor, twin critics, target networks, and optimiser state. It keeps
the original V1 sensor-only training contract unchanged: the `0.025`
observation-noise profile, `3e-4` actor and critic learning rates, `0.1`
training action noise, TD3 delay `2`, and one gradient batch every four
interactions. It collects fresh replay using the same held launch-biased random
profile used by the original pace runs. It adds no racing line, teacher,
curriculum, rollback, reward change, or fine-tuning override.

Its outputs are isolated from the base Agent 8 model:

```text
models/agent8_sensor_n_step_td3_v1_continuation/
models/training_runs/agent8_sensor_n_step_td3_v1_continuation/
```

Start a 100,000-step continuation:

```powershell
python scripts\train_sensor_n_step_td3_v1_continuation_agent.py --total-timesteps 100000 --seed 41 --evaluation-interval 10000 --evaluation-repeats 3 --checkpoint-interval 10000 --no-tensorboard
```

Evaluate the best continuation candidate:

```powershell
python scripts\evaluate_sensor_n_step_td3_v1_continuation_agent.py --repeats 10 --seed 50000
```

## Stability continuation

The separate stability continuation begins from the frozen V1 pace frontier,
`models/agent8_sensor_n_step_td3_v1_continuation/best_pace.pt`. It preserves
the complete actor-critic state and the existing sensor-only reward, action,
and observation contracts. It only reduces the actor learning rate to `3e-6`,
exploration noise to `0.01`, and the actor update cadence through TD3 delay `4`
with gradient batches every eight interactions. Fresh replay is collected by
the frozen policy before 2,000 critic-only updates; no racing line, teacher,
rollback, or reward modification is introduced.

Candidate selection remains passive and completion-first: completion rate,
then median completed-lap time, then distance. The pace checkpoint is never
overwritten. Results are written separately beneath:

```text
models/agent8_sensor_n_step_td3_v1_stability/
models/training_runs/agent8_sensor_n_step_td3_v1_stability/
data/evaluation/agent8_sensor_n_step_td3_v1_stability/
```

Run a short stability validation:

```powershell
python scripts\train_sensor_n_step_td3_v1_stability_agent.py --total-timesteps 50000 --seed 41 --evaluation-interval 10000 --evaluation-repeats 3 --checkpoint-interval 10000 --no-tensorboard
```

Evaluate the resulting candidate over 20 runs:

```powershell
python scripts\evaluate_sensor_n_step_td3_v1_stability_agent.py --repeats 20 --seed 50000
```

## Clean-reliability continuation

The stability pace frontier is intentionally not used by the application: it
only completes occasionally when its observations are perturbed, while the GUI
uses unperturbed TORCS telemetry. The clean-reliability continuation transfers
only that actor into a fresh critic and replay buffer, then trains and evaluates
solely on the exact zero-noise observation contract used by the application.

It retains the sensor-only reward, signed steering/longitudinal action, and
reward-only N-step TD3 algorithm. It adds no racing line, teacher, imitation,
curriculum, rollback, or driving intervention. The actor learning rate remains
conservative at `3e-6`, while TD3 uses the existing delayed actor updates and
fresh replay/critic warm-up.

The profile saves an application candidate only when a clean ten-run evaluation
achieves at least 8/10 completed laps and a median completed-lap time below
90 seconds. Until then, Agent 8 continues to load the existing default policy;
when a candidate passes, the GUI automatically prefers its
`best_evaluation.pt` checkpoint.

Outputs remain isolated:

```text
models/agent8_sensor_n_step_td3_v1_clean_reliability/
models/training_runs/agent8_sensor_n_step_td3_v1_clean_reliability/
data/evaluation/agent8_sensor_n_step_td3_v1_clean_reliability/
```

Start the clean-reliability run:

```powershell
python scripts\train_sensor_n_step_td3_v1_clean_reliability_agent.py --total-timesteps 300000 --seed 61 --evaluation-interval 10000 --evaluation-repeats 10 --checkpoint-interval 10000 --no-tensorboard
```

Evaluate the promoted application candidate:

```powershell
python scripts\evaluate_sensor_n_step_td3_v1_clean_reliability_agent.py --repeats 20 --seed 70000
```

## Robust-pace continuation

The clean-only reliability run showed that the `82.294 s` pace actor had not
seen enough of the observation conditions that produce its rare lap finishes.
The robust-pace experiment transfers that **Agent 8** actor alone into fresh
critics and fresh replay. It changes neither the reward, the controls, nor the
sensor-only observation features.

Training episodes are selected once per reset: 75% use the zero-noise GUI
contract and 25% use the original `0.025` feature-noise contract that produced
the pace frontier. Every baseline, scheduled evaluation, external evaluation,
and application race remains zero-noise. The actor rate is `3e-6`; replay is
collected with the frozen actor, then critics warm up before actor updates.

The profile also keeps a bounded, persistent elite replay containing complete
N-step transition traces from its own clean sub-90-second training laps. These
traces are captured only with zero observation noise, zero off-track steps, and
zero damage. Once available, at most 25% of a TD3 batch is drawn from the elite
buffer and at least 75% remains ordinary recent replay. The TD3 losses are
unchanged: there is no action copying, behaviour-cloning loss, teacher, or
open-loop playback. Every qualifying lap also saves the policy that existed at
discovery time plus JSON metadata beneath `elite_laps/`. The shared elite replay
is written atomically to `elite_success_replay.pt` and is loaded automatically
by later robust-pace runs.

Each scheduled evaluation rollout is a cold TORCS restart rather than a reset
within the current simulator process. Its exact policy is saved immediately,
before any later gradient update, beneath the run's
`evaluation_candidates/` directory with JSON score metadata. These are
discovery artefacts only: external evaluation decides whether any candidate is
reliable enough for promotion.

No racing line, target speed, teacher action, imitation, rollback, curriculum,
or control rule is introduced. A candidate reaches the application only when
its clean ten-run evaluation completes at least 8/10 laps with a median below
90 seconds. Its outputs remain isolated:

```text
models/agent8_sensor_n_step_td3_v1_robust_pace/
models/agent8_sensor_n_step_td3_v1_robust_pace/elite_laps/
models/training_runs/agent8_sensor_n_step_td3_v1_robust_pace/
data/evaluation/agent8_sensor_n_step_td3_v1_robust_pace/
```

Run a short validation before any longer run:

```powershell
python scripts\train_sensor_n_step_td3_v1_robust_pace_agent.py --total-timesteps 50000 --seed 71 --evaluation-interval 25000 --evaluation-repeats 10 --checkpoint-interval 25000 --no-tensorboard
```

Evaluate a resulting candidate under the GUI observation contract:

```powershell
python scripts\evaluate_sensor_n_step_td3_v1_robust_pace_agent.py --repeats 20 --seed 70000
```

## Self-imitation stability profile

This isolated profile addresses the rare-fast-lap reliability problem without
altering any existing Agent 8 checkpoint. It transfers only the actor from
`champion_82_294s_5of20_external.pt`, creates fresh critics and ordinary
replay, and migrates Agent 8's own recorded 83.620-second transition trace into
a versioned terminal-lap reward contract.

The critic batch is fixed at 80% ordinary experience and 20% self-generated
elite-lap experience. The actor objective adds a decaying mean-squared action
loss on only those elite rows. Its coefficient begins at `1.0`, decays to
`0.05` over 10,000 actor updates, and leaves the ordinary TD3 actor objective
active. This is accurately classified as self-imitation TD3, not reward-only
TD3: the demonstration was generated by Agent 8 itself, with no external
teacher or racing-line data.

The stability reward contains normalized forward velocity, a small quadratic
centre cost (`0.05`), a small squared steering-change cost (`0.0025`), a
failure penalty (`-10`), and a terminal lap bonus (`+1000`). A heading error
over 45 degrees terminates the episode. TD3 target-policy smoothing remains at
the existing configured values; evaluation remains deterministic with no
exploration noise.

Deterministic evaluation and application deployment use a steering EMA with
retention `0.8`. The filter resets for every episode and does not alter the
longitudinal action. Training executes the actor's exploratory actions directly
so the replay buffer represents what TORCS actually received.

The profile accepts either clean observations (`--observation-noise-std 0`) or
the torcsRL reference disturbance profile (`--observation-noise-std 0.025`).
Replay collection, training, and passive internal evaluation use the same
configured observation-noise level, avoiding a clean-training/noisy-evaluation
distribution mismatch.

Outputs remain isolated:

```text
models/agent8_sensor_n_step_td3_self_imitation_stability/
models/training_runs/agent8_sensor_n_step_td3_self_imitation_stability/
data/evaluation/agent8_sensor_n_step_td3_self_imitation_stability/
```

Run a short noise-aligned robustness validation from a protected pace actor:

```powershell
python scripts\train_sensor_n_step_td3_self_imitation_agent.py --resume models\agent8_sensor_n_step_td3_self_imitation_stability\pace_83_038s_clean_05pct_noisy.pt --total-timesteps 20000 --seed 74 --observation-noise-std 0.025 --fine-tune-actor-learning-rate 3e-7 --train-frequency 32 --evaluation-interval 5000 --evaluation-repeats 10 --checkpoint-interval 5000 --no-tensorboard
```

Evaluate a selected candidate with deterministic deployment smoothing and the
same observation-disturbance profile:

```powershell
python scripts\evaluate_sensor_n_step_td3_self_imitation_agent.py --policy-path <selected-candidate.pt> --repeats 20 --seed 82000 --observation-noise-std 0.025
```

The new profile is deliberately not added to Agent 8's default application
checkpoint list until an external evaluation demonstrates improved reliability.

## Recorded elite-lap demonstration

Console option `9` and the GUI agent named
`Agent 8 Recorded 83.620s Demonstration` replay the exact 4,040 raw actions
captured from Agent 8's clean 83.620-second training lap. Steering and the
signed throttle/brake command are replayed in order, while the existing live
RPM-based automatic gearbox remains active.

The portable recording is stored in
`data/recorded_laps/agent8_elite_83_620s_actions.npz`, with provenance and lap
metrics in `data/recorded_laps/agent8_elite_83_620s.json`. It does not depend on
an ignored training-run or model directory, so the demonstration works from a
fresh repository checkout.

This option is deliberately registered as
`agent8_recorded_elite_lap`, not as the neural Agent 8 policy. Its saved race
configuration sets `evaluation_eligible` and `neural_policy_active` to false.
It is an open-loop visualisation of a trajectory discovered by reward-only TD3,
so small differences in initial simulator state can make it diverge and its
result must not be reported as a DRL policy evaluation.
