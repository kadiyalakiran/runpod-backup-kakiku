# runpod-backup-kakiku

## What this is
Kiran's personal agentic AI development environment: coursework
(Stanford CS329A, Self-Improving AI Agents), reusable agent architecture
code, and applied projects — built toward becoming a professional agentic
AI developer/deployer, not just completing a course.

Runs on ephemeral RunPod pods with no persistent network volume — git is
the only thing that survives a pod swap. Everything here is written with
that constraint in mind.

## Structure
- `core/` — reusable agentic building blocks: agents/, verifiers/,
  planning/, rl/, eval/. Nothing starts here — code graduates in only
  after proving itself in a real course exercise or project. See
  core/CLAUDE.md for the promotion rule.
- `courses/` — coursework-derived reference material and runnable stubs
  (courses/cs329a — CS329A lecture notes + stub code, one folder per
  lecture, see courses/cs329a/CURRICULUM.md).
- `projects/` — actual applied work, one folder per project.
- `experiments/` — short-lived scratch work, not meant to last.

## Agentic AI design conventions
This repo follows a small set of design principles drawn from CS329A,
applied consistently across core/, courses/, and projects/:

- **generate → verify → improve is the default shape.** Most agent code
  here is a variant of that loop (STaR-style self-training, verifier-ranked
  selection, RL advantage estimation, multi-agent debate). When adding a
  new capability, ask which part of that loop it is before writing it.
- **Verification is the bottleneck, not generation.** When building
  anything self-improving, spend the real design effort on how correctness
  gets checked (a verifier, an eval harness, a ground-truth check) — that's
  what determines whether the loop actually compounds or just plateaus.
- **Pluggable model interfaces, always.** Agent/algorithm code takes a
  plain function (`llm_call(prompt) -> str`, `sample_fn`, `score_fn`, etc.)
  rather than importing a specific client. `llm_client.py` at the repo
  root is the one place that knows how to actually reach a model — local
  Ollama or RunPod-hosted, switched via `LLM_BASE_URL`/`LLM_MODEL` in
  `.env`. Never hardcode a base_url or model name anywhere else.
- **Every stub keeps a toy/fake demo.** The `if __name__ == "__main__":`
  block in each stub should always be runnable with zero external
  dependencies as a smoke test. When wiring in a real model, add a second,
  gated path (see courses/cs329a/lecture04_feedback_tools_code/
  react_agent.py's `USE_REAL_LLM` pattern) — don't replace the toy demo.

## Common commands
- `bash bootstrap.sh` — sets up venv, Claude Code CLI, VS Code extensions
  on a fresh pod. Idempotent, safe to re-run.
- `source /workspace/.venv/bin/activate` — activate the venv.
- `python3 llm_client.py` — check which LLM backend `.env` currently
  points at before debugging anything that calls a model.
- `git config --global user.email "..."` / `user.name "..."` — needed
  once per fresh pod before the first commit there.

## Dependencies
- Whenever you (Claude) install a new package — pip, npm, or otherwise —
  add it to the matching manifest (requirements.txt, package.json, etc.)
  in the SAME turn, before moving on. Never leave a package
  installed-but-untracked: there's no persistent volume on this setup, so
  anything only in the venv/node_modules is gone the moment this pod is
  replaced. The manifest file is the only thing that survives.
- Keep requirements.txt to top-level packages only, not a full
  `pip freeze`, unless a specific pinned version is actually load-bearing
  (e.g. matching a model's required transformers version).
- After adding to the manifest, remind the user to commit if you can't
  commit yourself, or just commit it as part of your change if you're
  already making a commit that turn.

## Do not
- Commit `.env` (real secrets) — only `.env.example` is tracked.
- Put model weights or large binaries in git — see .gitignore.
- Assume state from a previous pod persists. If it isn't in this repo,
  it doesn't exist on the next pod.
