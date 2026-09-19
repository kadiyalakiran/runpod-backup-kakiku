# Lecture 5 — Planning and Multi-Step Reasoning

## LATS (Language Agent Tree Search)
The lecture's central paper: unifies reasoning, acting, and planning by bringing
classic MCTS (Monte Carlo Tree Search) into the LLM reasoning path. The worked
example is a trip-planning prompt: reasoning steps propose sub-goals, actions
execute/look things up, and the tree search balances exploring new branches vs.
exploiting ones that already look promising.

Two things score a state in the tree, per the transcript:
1. A **value estimate** for that state (how good does this partial plan look).
2. A **self-consistency score** — sample multiple continuations from a state and
   measure how much they agree; high agreement across independent samples is
   itself evidence the state is a sound one to keep expanding. (Worked example:
   a state where 75% of sampled continuations agreed scored higher than one with
   lower agreement, even before checking correctness directly.)

This directly reuses ideas from lecture 2 (self-consistency, best-of-N) and
lecture 3 (verifier-style scoring) but applies them *per-node* in a search tree
rather than once at the end — the planning lecture is really "beam search /
best-of-N, but the branching happens over multi-step trajectories instead of
single answers."

## Interleaving planning and execution
A later point in the lecture: rather than fully planning then fully executing,
some agents interleave the two — plan a bit, execute a bit, replan based on what
came back — and parallel execution of multiple candidate plans is raised as an
open direction (using something like GPT-4o purely to *annotate* which part of a
trajectory was "planning" vs. "execution" for analysis, not as part of the agent
itself).

## Papers/artifacts referenced
- Zhou et al. (ICML), "Language Agent Tree Search Unifies Reasoning, Acting, and
  Planning in Language Models" (LATS).
