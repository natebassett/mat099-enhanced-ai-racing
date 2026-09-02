# Results and Evidence Workspace

The Results page is a read-only view of the experiment evidence already stored by
the project. It does not run TORCS, modify checkpoints, promote policies, or
rewrite evaluation and training logs.

## Views

### Overview

- Leads with laps completed, typical lap time, fastest lap, and trial count.
- Selects the fastest reliable representative from evaluations with at least ten
  trials and at least 80% completion.
- Compares Agents 6, 7, and 8 using direct completion counts and familiar
  percentage bars.
- Explains the difference between demonstrated pace and repeated reliability in
  plain language.

The overview is designed for AI novices. It does not expose checkpoint names,
contract identifiers, or raw source paths.

### Learning Journey

- Reads Agent 7 and Agent 8 `episodes.csv` files.
- Excludes evaluation rows from the training interaction count.
- Plots raw episode distance and the running best as separate series.
- Plots completed training-lap times against a visible 90-second reference.
- Summarises the selected session in a short plain-language result.

The raw series remains visible because a running-best curve alone hides policy
regression. A fast training episode is evidence of capability, not by itself
evidence of reliability.

### Agent Comparison

- Aggregates saved GUI race runs by agent and track.
- Compares best completed lap and observed completion rate.
- Reports sample counts and medians in the evidence table.

These are application race records, not controlled multi-seed evaluations. They
are useful for demonstrations and agent-to-agent comparison, while the planned
Research Evidence area remains the stronger source for reliability claims.

### Research Evidence

The detailed evaluation panel remains implemented as a self-contained view for the
planned **Settings > Research Evidence** area. It keeps every evaluation batch
separate, displays raw trial outcomes, and records the policy path plus observation,
action, and reward contract versions. Moving it out of the main Results flow keeps
the novice experience clear without discarding dissertation provenance.

## Interpretation Rules

1. Always report the number of evaluation trials with a completion percentage.
2. Use median completed-lap time for typical successful pace and fastest lap only
   as a capability result.
3. Keep failures in the denominator; do not calculate reliability from successful
   laps alone.
4. Compare policies on the same track and disclose differing model contracts.
5. Cite the exact source file shown beneath the relevant plot or table.

## Research Basis

The workspace follows the reporting concerns raised by:

- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical
  Precipice*: multiple runs, interval-aware reporting, and resistance to conclusions
  based on point estimates alone. <https://arxiv.org/abs/2108.13264>
- Henderson et al., *Deep Reinforcement Learning That Matters*: sensitivity to
  implementation, environment, hyperparameters, and evaluation protocol.
  <https://arxiv.org/abs/1709.06560>
- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic
  Methods*: the source algorithm for TD3. <https://arxiv.org/abs/1802.09477>

The interface does not claim formal confidence intervals where the saved evidence
does not support them. Trial counts, failures, provenance, and contract versions
remain visible so dissertation figures can be interpreted in context.
