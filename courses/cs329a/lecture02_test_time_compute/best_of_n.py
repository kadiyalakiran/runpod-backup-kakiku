"""
Lecture 2 — Test-Time Compute Scaling: Best-of-N strategies.

Three selection strategies over N sampled candidates, from cheapest to most
structured. None of this calls a real model — wire in your own `sample_fn`
(e.g. hitting a RunPod-hosted Qwen endpoint) to use it for real.

    sample_fn(prompt: str) -> str
        Returns one sampled completion. Called N times independently.

    score_fn(prompt: str, candidate: str) -> float
        A verifier score in [0, 1] (see ../lecture03_robust_verification/verifiers.py
        for ORM/PRM implementations to plug in here).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

SampleFn = Callable[[str], str]
ScoreFn = Callable[[str, str], float]
AnswerExtractFn = Callable[[str], str]


def majority_vote(
    prompt: str,
    sample_fn: SampleFn,
    n: int = 16,
    extract_answer: AnswerExtractFn = lambda x: x.strip(),
) -> tuple[str, dict]:
    """Sample N times, return the most common final answer.

    Cheap and effective when correct answers are the "canonical" simple ones —
    per the lecture, this plateaus after ~10-50 samples and degrades on harder
    problems where wrong answers cluster too. Use verifier_rank below once you
    notice that plateau on your own task.
    """
    candidates = [sample_fn(prompt) for _ in range(n)]
    answers = [extract_answer(c) for c in candidates]
    counts = Counter(answers)
    best_answer, votes = counts.most_common(1)[0]
    return best_answer, {"votes": votes, "n": n, "distribution": dict(counts)}


def verifier_rank(
    prompt: str,
    sample_fn: SampleFn,
    score_fn: ScoreFn,
    n: int = 16,
) -> tuple[str, dict]:
    """Sample N candidates, score each with a verifier, return the top-scored one.

    This is the "model-based ranker" strategy from the lecture — it's what you
    reach for once majority voting plateaus. score_fn is any ORM/PRM-style
    verifier; see lecture03 for two concrete implementations.
    """
    candidates = [sample_fn(prompt) for _ in range(n)]
    scored = [(c, score_fn(prompt, c)) for c in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    best_candidate, best_score = scored[0]
    return best_candidate, {"score": best_score, "n": n, "all_scores": [s for _, s in scored]}


@dataclass
class BeamNode:
    partial: str
    score: float


def beam_search_with_prm_pruning(
    prompt: str,
    expand_fn: Callable[[str], Sequence[str]],
    step_score_fn: Callable[[str, str], float],
    beam_width: int = 4,
    max_steps: int = 8,
) -> BeamNode:
    """Beam search over partial reasoning traces, pruned by a step-level verifier.

    expand_fn(partial) -> candidate continuations for one more reasoning step.
    step_score_fn(prompt, partial) -> a PRM-style score for the partial trace so far.

    Unlike best-of-N (which generates full candidates independently, then picks),
    this prunes low-scoring branches *during* generation — the lecture's point
    about not wasting compute extending reasoning that's already going wrong.
    """
    beam = [BeamNode(partial="", score=1.0)]
    for _ in range(max_steps):
        candidates: list[BeamNode] = []
        for node in beam:
            for continuation in expand_fn(node.partial):
                new_partial = node.partial + continuation
                score = step_score_fn(prompt, new_partial)
                candidates.append(BeamNode(partial=new_partial, score=score))
        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_width]
        if not beam:
            break
    return max(beam, key=lambda n: n.score)


if __name__ == "__main__":
    # Toy demo with a fake sample_fn so this runs with no dependencies.
    import random

    def fake_sample_fn(prompt: str) -> str:
        return random.choice(["42", "42", "17", "42", "6"])

    answer, info = majority_vote("What is 6 * 7?", fake_sample_fn, n=10)
    print("Majority vote answer:", answer, info)
