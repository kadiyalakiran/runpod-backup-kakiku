"""
Lecture 5 — LATS-style planner: MCTS over LLM reasoning/action states, scored by
a value estimate combined with a self-consistency score.

This is a simplified single-process MCTS, not the full LATS paper (no proper
backpropagation of values up the tree, no reflection-on-failure step) — it's
meant as a skeleton you extend once you have a real expand_fn/value_fn from
your own model, not a paper reproduction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

ExpandFn = Callable[[str], Sequence[str]]  # state -> candidate next-step continuations
ValueFn = Callable[[str], float]  # state -> estimated value in [0, 1]
SampleContinuationFn = Callable[[str], str]  # state -> one sampled continuation (for self-consistency)


@dataclass
class Node:
    state: str
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def ucb(self, exploration: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return self.value + exploration * math.sqrt(math.log(parent_visits + 1) / self.visits)


def self_consistency_score(
    state: str, sample_continuation: SampleContinuationFn, n: int = 5
) -> float:
    """Sample n continuations from this state and measure agreement — per the
    lecture, a state where most independent samples agree is itself evidence
    the state is worth expanding further, even before checking correctness."""
    samples = [sample_continuation(state) for _ in range(n)]
    if not samples:
        return 0.0
    most_common = max(set(samples), key=samples.count)
    return samples.count(most_common) / len(samples)


class LATSPlanner:
    def __init__(
        self,
        expand_fn: ExpandFn,
        value_fn: ValueFn,
        sample_continuation: SampleContinuationFn,
        is_terminal: Callable[[str], bool] = lambda s: False,
        consistency_weight: float = 0.3,
    ):
        self.expand_fn = expand_fn
        self.value_fn = value_fn
        self.sample_continuation = sample_continuation
        self.is_terminal = is_terminal
        self.consistency_weight = consistency_weight

    def score_state(self, state: str) -> float:
        """Combined score: model's own value estimate, blended with how much
        independent samples from this state agree with each other."""
        v = self.value_fn(state)
        c = self_consistency_score(state, self.sample_continuation)
        return (1 - self.consistency_weight) * v + self.consistency_weight * c

    def search(self, root_state: str, n_iterations: int = 20) -> Node:
        root = Node(state=root_state)
        for _ in range(n_iterations):
            node = self._select(root)
            if not self.is_terminal(node.state):
                node = self._expand(node)
            reward = self.score_state(node.state)
            self._backpropagate(node, reward)
        return root

    def _select(self, node: Node) -> Node:
        while node.children:
            node = max(node.children, key=lambda c: c.ucb())
        return node

    def _expand(self, node: Node) -> Node:
        continuations = self.expand_fn(node.state)
        if not continuations:
            return node
        for c in continuations:
            node.children.append(Node(state=node.state + c, parent=node))
        return random.choice(node.children)

    def _backpropagate(self, node: Node, reward: float) -> None:
        current: Node | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def best_path(self, root: Node) -> list[str]:
        path = [root.state]
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            path.append(node.state)
        return path


if __name__ == "__main__":
    def fake_expand(state: str) -> list[str]:
        if len(state) > 30:
            return []
        return [" step_a", " step_b"]

    def fake_value(state: str) -> float:
        return len(state) / 40.0

    def fake_sample_continuation(state: str) -> str:
        return random.choice(["step_a", "step_a", "step_b"])

    planner = LATSPlanner(
        expand_fn=fake_expand,
        value_fn=fake_value,
        sample_continuation=fake_sample_continuation,
    )
    root = planner.search("goal: plan a trip", n_iterations=15)
    print("Best path:", planner.best_path(root))
