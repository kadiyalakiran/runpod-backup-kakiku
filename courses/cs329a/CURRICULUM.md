# CS329A — Self-Improving AI Agents (Stanford, Mirhoseini & Chowdhery)

Notes and starter code, one folder per lecture, built from the lecture transcripts.
Each `notes.md` is a condensed, paraphrased summary — not a transcript copy — with
pointers to the underlying papers. Each stub is a minimal, runnable skeleton for the
core algorithm of that lecture; none are production-ready, and none call a real LLM
by default (they take a pluggable `llm_call` / `sample_fn` so you can wire in your
own RunPod-hosted model, e.g. the Qwen setup in `projects/ai-test`).

| # | Lecture | Core technique(s) | Stub |
|---|---|---|---|
| 01 | Course Overview | Scaling laws (compute/data/params) | notes only |
| 02 | Test-Time Compute Scaling | Best-of-N, majority vote, beam search + PRM pruning | `best_of_n.py` |
| 03 | Robust Verification | ORM vs. PRM, verifier-guided search | `verifiers.py` |
| 04 | Feedback with Tools & Code | STaR, ReAct, RLEF, Constitutional AI | `star_loop.py`, `react_agent.py` |
| 05 | Planning & Multi-Step Reasoning | LATS (MCTS + self-consistency) | `lats_planner.py` |
| 06 | Train-Time Scaling & RL | PPO → GRPO → DAPO | `rl_algorithms.py` |
| 07 | Self-Improvement & Deep Research | Search + calibration/uncertainty | `deep_research_agent.py` |
| 08 | Agentic Evaluations & Long-Horizon | METR time-horizon, GDPval, DeepScholar-bench style eval | `eval_harness.py` |
| 09 | Future Research Areas | Multi-agent debate, curriculum, open-endedness | `multi_agent_debate.py` |

## How to use this

- Treat `notes.md` as the "why" and the stub as the "what would I actually build."
- Promote anything that turns out generally useful into `../core/` (outside this
  folder) rather than letting it calcify inside a single lecture's directory —
  that's the point of keeping coursework and reusable architecture separate.
- The two throughlines of the whole course, per the transcripts: **verifiers**
  (lecture 3, but referenced in 8 of 9 lectures) and **STaR-style self-training
  loops** (lecture 6, referenced in 8 of 9 lectures). If you only build two things
  well from this course, build those two.
