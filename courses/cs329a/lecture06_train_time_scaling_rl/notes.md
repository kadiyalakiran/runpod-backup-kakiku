# Lecture 6 — Train-Time Scaling and Scaling RL

This is the deepest lecture in the course (STaR alone gets 49 mentions here) and
the one that ties test-time tricks (lectures 2–5) into actual weight updates.

## STaR, revisited with more detail
Same algorithm as lecture 4, but here the lecture pushes on its limits: on a math
benchmark, STaR-style fine-tuning boosted accuracy noticeably, but improvement
plateaus after a few iterations — the transcript explicitly compares this to RL's
own plateau problems ("RL has its own set of challenges but here it also starts
to plateau"). Two follow-ons mentioned: a paper training a **generator and
verifier together in a loop** (verifier bootstrapping feeding back into generator
training, not just a one-off filter), and **Quiet-STaR**, which adds an internal
"thinking" step so the model rationalizes *before* answering rather than only
when corrected. Also notable: starting self-improvement from a model already
strong in one verifiable domain (e.g. a code-tuned base model) transfers — models
that started from a coding-focused checkpoint picked up math self-improvement
faster than ones that didn't.

## PPO → GRPO → DAPO: the actual RL lineage
- **PPO**: needs a policy, an old policy (for the clip ratio), and a separate
  critic/value network to estimate advantage. Memory-heavy — you're holding
  multiple full copies of the model.
- **GRPO** (Group Relative Policy Optimization): drops the critic entirely.
  Instead, sample a *group* of completions for the same prompt, and use the
  group's own reward distribution as the baseline — each completion's advantage
  is its score relative to the group mean/std. No value network, much less
  memory, which per the lecture is what actually made scaling RL up to a 32B
  model (Qwen2.5-32B) practical at course-compute budgets. On a math benchmark
  they cite moving from 46.8 to 51.7 with this shift.
- **DAPO** (built on top of GRPO, tested on Qwen-32B on AIME): fixes several
  concrete failure modes that show up once you naively scale GRPO up:
  - **Clip-higher / asymmetric clipping**: standard PPO/GRPO clipping is
    symmetric, which caps how much a token's probability can *increase* just as
    much as how much it can decrease — this collapses exploration on
    already-high-probability tokens. Allowing bigger *upward* moves (asymmetric
    clipping) measurably raised accuracy in their ablation.
  - **Dynamic/overlong filtering**: filter out degenerate long sequences before
    they pollute the batch. Cited as a 30 → 36 accuracy jump on its own, on the
    AIME-style benchmark, before adding asymmetric clipping on top.
  - **Token-level loss** instead of sample-level loss aggregation.
  - Explicit handling for the case where every sample in a group gets the same
    reward (advantage would otherwise be exactly zero and give no learning
    signal).

## Papers/artifacts referenced
- Zelikman et al., "STaR" (and the Quiet-STaR follow-on).
- Shao et al. (DeepSeekMath), "GRPO."
- The DAPO paper (Decoupled Clip and Dynamic sAmpling Policy Optimization),
  evaluated on Qwen2.5-32B on AIME.
