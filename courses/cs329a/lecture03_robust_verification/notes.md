# Lecture 3 — Robust Verification

This is the most heavily-covered single topic in the transcripts (101 mentions of
"verifier" in this lecture alone) — the instructors are explicit that robust
verification, not generation, is the actual bottleneck on how far self-improvement
loops can run: in verifiable domains (math, code) verifiers are cheap and the
generate→verify→retrain flywheel compounds; in domains without cheap verification
(creative writing, open-ended tasks), the flywheel stalls.

## ORM vs. PRM
- **ORM (Outcome Reward Model)**: trained on (generated solution, final-answer
  correctness) pairs. One label per full trajectory. Cheap to collect labels for
  (just check the final answer against ground truth).
- **PRM (Process Reward Model)**: trained on step-by-step correctness labels
  within a solution — a score per reasoning step, aggregated as a product of
  per-step probabilities for the whole trace. Much more expensive to label
  (originally human-annotated, e.g. OpenAI's PRM800K), but per the lecture,
  PRM outperforms both ORM and majority voting, is more data-efficient
  (needs fewer labels to reach the same accuracy as ORM), and — notably —
  detects *correct process, wrong final answer* cases that ORM structurally
  can't see.
- Later PRM work iterates the labels themselves: use an LLM to judge each step,
  collect that as training data, retrain the PRM, repeat — bootstrapping the
  verifier the same way STaR bootstraps the generator (lecture 6).

## Generator/verifier size tradeoff
The lecture covers an ablation: larger generator + smaller verifier vs. smaller
generator + larger verifier. Larger generators benefit more from verification.
There's also a diminishing-returns point on sample count — beyond roughly 400–800
sampled solutions, adding more samples for the verifier to rank stops helping,
because the verifier itself becomes the bottleneck at that point, not the
generator's coverage.

## Papers/artifacts referenced
- Cobbe et al., "Training Verifiers to Solve Math Word Problems" (the original
  ORM-style verifier on GSM8K).
- Lightman et al., "Let's Verify Step by Step" (PRM800K, the ORM-vs-PRM comparison).
