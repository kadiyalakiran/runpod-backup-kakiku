# Lecture 4 — Learning from Feedback with Tools and Code

Four techniques, all variations on "the model gets better by acting on feedback
that isn't a human label."

## STaR (Self-Taught Reasoner) — first appearance; deep dive is lecture 6
Bare-bones version: start with a small seed set of (question, rationale, answer)
examples. Generate rationales for a larger training set. Filter to keep only the
ones whose final answer matches ground truth. If the answer is wrong, give the
model a **hint** (the correct answer) and have it rationalize backward to produce
a plausible-looking correct rationale anyway ("rationalization"). Fine-tune on the
filtered set. Repeat. Per the transcript: pure STaR performance plateaus after a
few iterations, and human raters preferred STaR-generated rationales over few-shot
prompted ones, but rationalization *without* iteration didn't help — the iterative
retraining is doing the real work, not the hint mechanism alone.

## ReAct (Reason + Act)
One of the first abstractions to combine chain-of-thought reasoning with tool use
in the same language stream: the model interleaves **Thought → Action →
Observation** steps, where Action is a call into an external action space (search,
a calculator, an API). Explicitly framed as letting the model "do this in language
space." Two failure modes called out directly: hallucination is the dominant
failure mode for ReAct without fine-tuning, and errors compound over a long
trajectory (an early wrong observation cascades). Fine-tuning on successful ReAct
trajectories closes much of that gap.

## RLEF (Grounding Code LLMs with Execution Feedback)
RL fine-tuning where the reward signal comes from actually running the generated
code and observing pass/fail. The lecture frames it as one of the first papers to
show that a model can learn, across turns of a multi-turn repair loop (turn 1 →
turn 2 → turn 3), to use execution feedback effectively to fix its own code —
i.e., the *skill of iterating* is itself learnable via RL, not just baked in via
prompting.

## Constitutional AI (Anthropic)
Two-stage self-critique loop, not a single trick:
1. **Supervised stage**: the model critiques and revises its own outputs against
   a small set of human-written principles (the lecture cites ~16 principles),
   producing a self-improved dataset to fine-tune on.
2. **RL stage**: use the model's own chain-of-thought critique reasoning as the
   preference signal for RL, instead of (or alongside) human preference labels.
Per the lecture's own comparison: pure "constitutional supervised learning" alone
underperforms the full pipeline — you need both stages, not just the self-critique
fine-tuning step, to match a fully human-feedback-based approach.

## Papers/artifacts referenced
- Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning"
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
- "RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning"
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback"
