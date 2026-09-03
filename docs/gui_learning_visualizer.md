# Agent Learning Visualizer

## Purpose

The Agent Lab learning visualizer explains how each project agent changes, or
does not change, its decision policy. It is designed for viewers who may not
have studied reinforcement learning while preserving the distinction between
an educational animation and measured experiment data.

## Supported views

- Agents 7 and 8 show the TD3 replay, actor, twin-critic, target, loss,
  gradient, and delayed-update sequence.
- Dyna-Q agents show state encoding, epsilon-greedy choice, TD error, Q-table
  update, and model replay.
- Map-aware, rule-based, and random agents show their decision path and state
  explicitly that no neural weights are trained.

The diagram uses a small number of representative nodes. The real Agent 7 and
Agent 8 actors contain 141 inputs and two hidden layers of 256 units; drawing
every connection would hide the learning sequence rather than clarify it.

## Evidence boundary

The animated gradient is a conceptual example of one update. It does not claim
to reproduce a historical gradient because per-update tensors were not saved
during training.

For TD3 agents, the checkpoint banner is measured data. The GUI loads only the
saved actor tensors and reports:

- actor parameter count;
- mean absolute weight;
- environment-step count when the checkpoint records it;
- actor-update count when the checkpoint records it.

Stable-Baselines3 `.zip` policies and the native Agent 7/8 `.pt` format are
both supported. Loading is read-only and uses PyTorch's weights-only mode. The
visualizer never writes to a policy or participates in training.

## Research basis

The update sequence follows the three defining TD3 mechanisms described by
Fujimoto, van Hoof, and Meger: clipped double-Q targets, delayed policy updates,
and target policy smoothing.

- Fujimoto, S., van Hoof, H., and Meger, D. (2018). *Addressing Function
  Approximation Error in Actor-Critic Methods*.
  https://arxiv.org/abs/1802.09477
- Olah, C. et al. (2018). *The Building Blocks of Interpretability*.
  https://distill.pub/2018/building-blocks/
- Wang, Z. et al. (2020). *CNN Explainer: Learning Convolutional Neural
  Networks with Interactive Visualization*.
  https://arxiv.org/abs/2004.15004

The latter two sources informed the use of progressive disclosure,
representative rather than exhaustive nodes, direct labels, and step-by-step
animation. They do not define or alter the racing algorithms.
