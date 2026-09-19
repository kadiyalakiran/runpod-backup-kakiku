# Lecture 1 — Course Overview

## Framing
The lecture opens with the standard pre-training scaling picture — loss falls
predictably as you increase compute, data, and parameter count (the trends behind
GPT-3 through the current generation of frontier models). That framing exists to
set up the course's actual thesis: pre-training scaling is not the only axis left.
The rest of the course is organized around a second axis — scaling what happens
*after* pre-training, at inference time and through self-generated training data —
which is why the course is called "self-improving agents" rather than "agents."

## The arc of the course (as stated)
1. Self-improvement techniques for a single model: verifiers, test-time compute,
   search combined with LLMs, RL at train time (lectures 2–6).
2. Augmenting LLMs with tools, code execution, and memory (lecture 4 and around).
3. Multi-step reasoning and planning for agentic workflows (lecture 5).
4. Building evaluation frameworks robust enough to trust the above (lecture 8).
5. Open problems — where the loop still breaks (lecture 9).

## Why this matters for your own agent-building work
The course's implicit claim is that "self-improvement" is not one technique — it's
a family of loops that all share the same shape: **generate → verify → use the
verified signal to get better next time**. What changes lecture to lecture is
*what* gets verified (a single answer, a multi-step trajectory, a code execution
result) and *when* the improvement happens (at inference time vs. baked into the
weights via RL). That shape is worth keeping in mind as an organizing principle
for `core/` as it grows — most of what you build here will be a variant of that
loop with a different verifier and a different action space.

## Papers/artifacts referenced
- No specific paper for this lecture — it's framing + logistics.

No stub for this lecture — it's scene-setting, not an algorithm.
