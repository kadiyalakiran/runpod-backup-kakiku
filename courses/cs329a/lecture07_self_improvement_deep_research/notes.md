# Lecture 7 — Self-Improvement and Deep Research Agents

## Building a deep research agent
The lecture's worked example is a search-augmented agent that iteratively issues
queries, retrieves documents, and refines its answer — "deep research" agents
built on top of the search + verification machinery from earlier lectures.

## Uncertainty as the control signal
The recurring theme is using the model's own **expressed uncertainty** to decide
when to search more, not just as a diagnostic:
- Knowledge gaps in a benchmark question tend to surface as uncertainty in the
  model's reasoning tokens *before* the final answer — i.e. uncertainty is
  legible in the trace if you look for it, not just in the final confidence.
- That uncertainty cascades: once the model is uncertain early in a chain, it
  propagates forward and shows up in the final answer's confidence too.
- Uncertainty can be measured at the level of the *search queries themselves*,
  not just the final answer — the transcript describes tracking uncertainty in
  the tokens used to generate a search query, separately from document-quality
  signals, and using that to decide whether another round of search is needed.
- A control loop follows naturally: retrieve → check relevance/quality of what
  came back → if uncertainty is still high, search again → stop once confidence
  is high enough or a budget is hit.

## Calibration is a separate problem from search quality
A second, related point: models tend to be **overconfident** by default —
calibration (does stated confidence match actual accuracy) is generally poor
without explicit intervention. RLHF-style fine-tuning is mentioned as one lever
to push a model toward better calibration, separate from anything about search.

## Papers/artifacts referenced
- No single named paper for the deep-research-agent architecture itself in this
  lecture (it's presented as a synthesis of earlier techniques); the calibration
  discussion connects to the general LLM-calibration literature (e.g. work on
  verbalized confidence and RLHF's effect on calibration).
