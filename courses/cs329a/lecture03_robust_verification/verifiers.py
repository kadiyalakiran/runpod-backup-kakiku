"""
Lecture 3 — Robust Verification: ORM and PRM verifier interfaces.

Both are stubs around a pluggable scoring model — swap `score_model` for a real
classifier/LLM call. The point of writing them side by side is the structural
difference the lecture emphasizes: ORM scores the *whole trajectory once*, PRM
scores *each step* and aggregates, which is what lets it catch a correct-process/
wrong-answer or wrong-process/lucky-answer case that ORM can't distinguish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


class StepSplitter(Protocol):
    def __call__(self, solution: str) -> list[str]:
        ...


def default_step_splitter(solution: str) -> list[str]:
    """Naive step split — replace with something that respects your model's
    actual reasoning delimiters (e.g. newline-separated steps, or explicit
    <step> tags if you control the generation format)."""
    return [s.strip() for s in solution.split("\n") if s.strip()]


@dataclass
class OutcomeRewardModel:
    """ORM: one correctness score for the whole (question, solution) pair.

    score_model(question, full_solution) -> probability the FINAL ANSWER is correct.
    Cheapest to train (one label per trajectory) but structurally blind to whether
    the reasoning that got there was sound.
    """

    score_model: Callable[[str, str], float]

    def score(self, question: str, solution: str) -> float:
        return self.score_model(question, solution)


@dataclass
class ProcessRewardModel:
    """PRM: a correctness score per reasoning step, aggregated across the trace.

    step_score_model(question, steps_so_far, next_step) -> probability that step
    is a valid continuation. Aggregation is the product of per-step probabilities,
    matching the lecture's description — a single bad step tanks the whole score,
    which is exactly the property that lets PRM catch a lucky-final-answer trace
    that took a wrong turn along the way.
    """

    step_score_model: Callable[[str, Sequence[str], str], float]
    step_splitter: StepSplitter = field(default=default_step_splitter)

    def score(self, question: str, solution: str) -> float:
        steps = self.step_splitter(solution)
        product = 1.0
        history: list[str] = []
        for step in steps:
            product *= self.step_score_model(question, history, step)
            history.append(step)
        return product

    def per_step_scores(self, question: str, solution: str) -> list[float]:
        """Useful for debugging — shows exactly where a trace goes wrong,
        which is the PRM's actual advantage over ORM."""
        steps = self.step_splitter(solution)
        scores = []
        history: list[str] = []
        for step in steps:
            s = self.step_score_model(question, history, step)
            scores.append(s)
            history.append(step)
        return scores


def bootstrap_prm_labels(
    question: str,
    solution_steps: Sequence[str],
    judge_fn: Callable[[str, Sequence[str], str], float],
) -> list[float]:
    """The PRM-bootstrapping move mentioned in the lecture: use an LLM (or an
    existing PRM) as the judge to generate step labels, rather than paying for
    human step-level annotation. Feed the output into a training set to fine-tune
    a cheaper/faster PRM — the verifier's own version of a self-improvement loop.
    """
    history: list[str] = []
    labels = []
    for step in solution_steps:
        labels.append(judge_fn(question, history, step))
        history.append(step)
    return labels


if __name__ == "__main__":
    def fake_orm(question: str, solution: str) -> float:
        return 0.9 if solution.strip().endswith("42") else 0.1

    def fake_step_model(question: str, history: list[str], step: str) -> float:
        return 0.95 if "error" not in step.lower() else 0.05

    orm = OutcomeRewardModel(score_model=fake_orm)
    prm = ProcessRewardModel(step_score_model=fake_step_model)

    solution = "Compute 6 * 7\nMultiply: 6 * 7 = 42\nFinal answer: 42"
    print("ORM score:", orm.score("What is 6*7?", solution))
    print("PRM score:", prm.score("What is 6*7?", solution))
    print("PRM per-step:", prm.per_step_scores("What is 6*7?", solution))
