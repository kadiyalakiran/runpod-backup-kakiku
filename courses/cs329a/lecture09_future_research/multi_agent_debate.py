"""
Lecture 9 — Multi-agent debate as a TRAINING DATA generator, not just an
inference-time trick. Two agents argue over multiple rounds; a critic contrasts
their positions each round; the resulting (position, critique) pairs are the
thing worth keeping — this is STaR's filter-and-keep move (lecture 4/6) applied
to disagreement instead of ground-truth correctness, which is why it's useful
in domains lecture 3 flags as lacking a cheap verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

DebaterFn = Callable[[str, list[str]], str]  # (question, prior_turns_from_both_sides) -> this turn's argument
CriticFn = Callable[[str, str, str], dict]  # (question, position_a, position_b) -> {"preferred": "a"|"b", "critique": str}


@dataclass
class DebateTurn:
    round_num: int
    position_a: str
    position_b: str
    critique: dict


@dataclass
class DebateTrajectory:
    question: str
    turns: list[DebateTurn] = field(default_factory=list)

    def as_training_example(self) -> dict:
        """The thing you'd actually keep for fine-tuning: the winning position
        from the final round, plus the critique that justified preferring it —
        a contrastive training pair with no ground-truth label required.
        """
        final = self.turns[-1]
        winner = final.position_a if final.critique["preferred"] == "a" else final.position_b
        loser = final.position_b if final.critique["preferred"] == "a" else final.position_a
        return {
            "question": self.question,
            "preferred": winner,
            "rejected": loser,
            "critique": final.critique["critique"],
            "n_rounds": len(self.turns),
        }


def run_debate(
    question: str,
    debater_a: DebaterFn,
    debater_b: DebaterFn,
    critic: CriticFn,
    n_rounds: int = 3,
) -> DebateTrajectory:
    trajectory = DebateTrajectory(question=question)
    history_a: list[str] = []
    history_b: list[str] = []

    for round_num in range(n_rounds):
        position_a = debater_a(question, history_a + history_b)
        position_b = debater_b(question, history_a + history_b)
        history_a.append(position_a)
        history_b.append(position_b)

        critique = critic(question, position_a, position_b)
        trajectory.turns.append(
            DebateTurn(round_num=round_num, position_a=position_a, position_b=position_b, critique=critique)
        )
    return trajectory


def build_contrastive_dataset(trajectories: list[DebateTrajectory]) -> list[dict]:
    """Run this over many debates to build the actual fine-tuning set — the
    lecture's point that debate can be a self-improvement DATA SOURCE, not
    just a better inference-time answer."""
    return [t.as_training_example() for t in trajectories]


if __name__ == "__main__":
    def fake_debater_a(question: str, history: list[str]) -> str:
        return f"[A, round {len(history)//2}] I argue the answer emphasizes cost."

    def fake_debater_b(question: str, history: list[str]) -> str:
        return f"[B, round {len(history)//2}] I argue the answer emphasizes reliability."

    def fake_critic(question: str, pos_a: str, pos_b: str) -> dict:
        # toy: alternate which side "wins" to show both branches
        preferred = "b" if "reliability" in pos_b else "a"
        return {"preferred": preferred, "critique": "Reliability argument better addresses the failure mode raised."}

    traj = run_debate("Should we optimize for cost or reliability?", fake_debater_a, fake_debater_b, fake_critic)
    example = traj.as_training_example()
    print("Preferred:", example["preferred"])
    print("Critique:", example["critique"])
    print("Rounds:", example["n_rounds"])
