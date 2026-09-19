# Lecture 9 — Future Research Areas

The synthesis/open-problems lecture. Three threads, each framed as unfinished
business the earlier lectures' techniques don't fully solve.

## Open-endedness
Explicitly framed as deserving "a whole class" on its own — the distinction the
lecture draws is between running *one* self-improvement loop well (everything in
lectures 2–7) versus building a system that keeps generating genuinely *new*
challenges for itself indefinitely, rather than converging and plateauing on a
fixed task distribution (the plateau problem raised repeatedly in lectures 4 and
6 around STaR and RL).

## Multi-agent debate as a training-data source
Not just an inference-time trick (get two agents to argue, take the better
answer) — the lecture frames debate as a way to **generate trajectories worth
training on**: run multiple rounds of debate, have a critic model learn to
contrast the arguments, and use the resulting contrastive trajectories as
training signal. This is effectively STaR's filtering move (lecture 4/6) applied
to *disagreement* as the filter instead of ground-truth correctness — useful
precisely in domains where you don't have a cheap verifier (creative,
subjective, or open-ended tasks — the exact gap lecture 3 identifies as where
the self-improvement flywheel stalls).

## Curriculum, tied to verification difficulty
The lecture connects curriculum design directly back to verification: you need
some way to judge difficulty in order to sequence a curriculum, and that
judgment is itself a verification problem. This ties lecture 9 back to lecture 3
as the actual bottleneck — curriculum design doesn't route around the
verification problem, it depends on it.

## Reward hacking, named directly as a risk
Called out explicitly: an agent slightly "off-base" can learn to game its reward
signal rather than solve the intended task — the standing risk of any of the RL
techniques in lecture 6, worth designing verifiers (lecture 3) defensively
against rather than assuming a high reward score means genuine task success.

## Papers/artifacts referenced
- No single paper anchors this lecture — it's a synthesis pointing at open
  problems in open-endedness, multi-agent training-data generation, curriculum
  learning, and reward hacking as a general RL failure mode.
