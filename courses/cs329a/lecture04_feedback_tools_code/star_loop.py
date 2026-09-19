"""
Lecture 4 / 6 — STaR: Self-Taught Reasoner.

Bare-bones loop per the transcript: generate rationales, keep the ones whose
final answer checks out, rationalize-with-a-hint the ones that don't, fine-tune
on the filtered set, repeat. This stub implements the DATA-GENERATION side of the
loop (which is the interesting, reusable part) and leaves fine-tuning as a
callback you'd wire to your actual training code — that part is infra-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

GenerateFn = Callable[[str], str]  # question -> rationale + answer
HintedGenerateFn = Callable[[str, str], str]  # (question, correct_answer) -> rationale + answer
ExtractAnswerFn = Callable[[str], str]
FineTuneFn = Callable[[list["Example"]], None]  # your actual training step


@dataclass
class Example:
    question: str
    rationale_and_answer: str
    was_rationalized: bool = False


@dataclass
class STaRLoop:
    generate: GenerateFn
    generate_with_hint: HintedGenerateFn
    extract_answer: ExtractAnswerFn
    fine_tune: FineTuneFn
    seed_examples: list[Example] = field(default_factory=list)

    def run_iteration(
        self, questions_and_answers: list[tuple[str, str]]
    ) -> tuple[list[Example], dict]:
        """One pass of generate -> filter -> rationalize-on-failure.

        questions_and_answers: (question, ground_truth_answer) pairs to attempt.
        Returns the filtered training set for this iteration plus stats, so you
        can watch for the plateau the transcript describes (accuracy gain per
        iteration shrinking, then flattening after a handful of rounds).
        """
        kept: list[Example] = []
        n_correct_first_try = 0
        n_rationalized = 0
        n_dropped = 0

        for question, ground_truth in questions_and_answers:
            output = self.generate(question)
            if self.extract_answer(output) == ground_truth:
                kept.append(Example(question, output, was_rationalized=False))
                n_correct_first_try += 1
                continue

            # Wrong on the first try: give the hint and let it rationalize backward.
            rationalized = self.generate_with_hint(question, ground_truth)
            if self.extract_answer(rationalized) == ground_truth:
                kept.append(Example(question, rationalized, was_rationalized=True))
                n_rationalized += 1
            else:
                # Even with the hint it can't produce a matching answer — drop it.
                # (In practice you might still keep the ground truth answer with
                # a templated rationale here; the transcript notes real STaR
                # implementations vary on this.)
                n_dropped += 1

        stats = {
            "n_correct_first_try": n_correct_first_try,
            "n_rationalized": n_rationalized,
            "n_dropped": n_dropped,
            "total": len(questions_and_answers),
        }
        return kept, stats

    def run(
        self,
        questions_and_answers: list[tuple[str, str]],
        n_iterations: int = 3,
    ) -> list[dict]:
        """Multiple rounds. Watch `history` for the plateau — the transcript is
        explicit that STaR-alone accuracy gains flatten after a few iterations,
        which is exactly the kind of thing worth plotting once this is wired to
        a real model instead of a fake generate_fn."""
        history = []
        examples = list(self.seed_examples)
        for i in range(n_iterations):
            new_examples, stats = self.run_iteration(questions_and_answers)
            examples.extend(new_examples)
            self.fine_tune(examples)  # your training step — no-op stub below
            stats["iteration"] = i
            history.append(stats)
        return history


if __name__ == "__main__":
    # Toy demo: "answer" is always the digit sum of the question length, to
    # give the fake generator something deterministic to sometimes get wrong.
    def fake_generate(question: str) -> str:
        # deliberately wrong most of the time to exercise the rationalize path
        return f"Reasoning... Final answer: {len(question) % 3}"

    def fake_generate_with_hint(question: str, correct: str) -> str:
        return f"Reasoning backward from the hint... Final answer: {correct}"

    def extract(output: str) -> str:
        return output.rsplit(":", 1)[-1].strip()

    def noop_fine_tune(examples: list[Example]) -> None:
        pass  # wire to your real training loop

    loop = STaRLoop(
        generate=fake_generate,
        generate_with_hint=fake_generate_with_hint,
        extract_answer=extract,
        fine_tune=noop_fine_tune,
    )
    data = [(f"question {i}", "1") for i in range(10)]
    for stats in loop.run(data, n_iterations=2):
        print(stats)
