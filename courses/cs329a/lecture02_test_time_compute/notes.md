# Lecture 2 — Test-Time Compute Scaling

## Core idea
Instead of (or in addition to) making the model bigger, spend more compute *per
query* at inference time: sample many candidate answers and pick a good one. The
lecture works through a spectrum of "picking" strategies, roughly in increasing
sophistication:

- **Majority voting**: sample N times, take the most common final answer. Cheap,
  but the transcript is explicit about where it breaks down — accuracy plateaus
  after roughly 10–50 samples, and the plateau is worse on harder problems, because
  the *wrong* answers stop being uniformly wrong and start clustering around a few
  plausible-but-incorrect answers just as much as the right one does. Majority
  voting only works well when correct answers are simple/canonical relative to the
  space of things the model might say.
- **Model-based rankers (verifiers)**: instead of counting votes, score each
  candidate with a learned verifier and take the best-scored one. This is the
  bridge into lecture 3.
- **Beam search + PRM pruning**: rather than generating N full candidates
  independently and picking at the end, use a process reward model to prune
  low-value branches *during* generation, so compute isn't wasted extending
  reasoning paths that were already going wrong.

## Why this is the right first stub to build
Best-of-N (in its majority-vote and verifier-ranked forms) is the simplest
instance of the course's generate→verify→select loop, and every later lecture's
technique is a more expensive/more structured version of the same move. Building
this one cleanly gives you the interface (`sample_fn`, `score_fn`, `select_fn`)
you'll reuse for the verifier and planning stubs.

## Papers/artifacts referenced
- General "best-of-N / self-consistency" line of work (Wang et al., self-consistency
  decoding) plus the process-reward-guided beam search framing that lecture 3
  develops further (Lightman et al., "Let's Verify Step by Step").
