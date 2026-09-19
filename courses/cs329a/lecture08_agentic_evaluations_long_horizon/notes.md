# Lecture 8 — Agentic Evaluations and Long-Horizon Tasks

## Why this lecture exists
Traditional chatbot-style benchmarks (single-turn Q&A) don't capture what
actually breaks in agents: work that unfolds over many steps, where errors
compound and the model has to sustain a plan over a long context. The lecture
walks through several benchmarks built specifically to measure *that*, not
single-shot accuracy.

## Benchmarks covered
- **A METR-style benchmark** (the transcript describes an OpenAI-adjacent
  benchmark measuring how long a task an autonomous model can complete,
  matching the well-known "time-horizon" framing — plot task duration a human
  would need against the length of task the model can complete unsupervised).
  Composed of ~170 tasks across three task suites, each with human time
  estimates so model performance can be expressed as "duration of task
  completed" rather than a bare accuracy number.
- **A live, Stanford-built benchmark** focused on safety-relevant long-horizon
  behavior — explicitly framed as trying to surface failure modes traditional
  chatbot benchmarks can't, including using models' own internal pull requests
  as one data source.
- **A "deep scholar"-style benchmark** (also Stanford), aimed specifically at
  evaluating deep-research-agent output quality — the lecture notes it still
  needs humans to evaluate the output, i.e. it isn't yet a fully automated
  metric.

## Recurring caveats the lecture is explicit about
- **No benchmark is perfect** — every one of these has known gaps, and the
  lecture spends time on what each benchmark specifically gets wrong, not just
  what it measures.
- **Saturation**: most benchmarks eventually saturate as models improve, which
  is part of why this area keeps producing new benchmarks rather than settling
  on one.
- **Failure-mode analysis matters more than the aggregate score** — the
  lecture pushes on looking at *which* categories of tasks fail and *why*
  (e.g. specific classes of "following instructions" errors), not just a
  single leaderboard number.

## Papers/artifacts referenced
- METR, "Measuring AI Ability to Complete Long Tasks" (the time-horizon framing).
- A Stanford-built agentic safety benchmark (name not fully captured in the
  transcript — verify against the course website's reading list).
- A Stanford "deep scholar"-style benchmark for research-agent evaluation
  (likely DeepScholar-Bench — verify against the course website).
