"""
Lecture 7 — Deep research agent, controlled by uncertainty rather than a fixed
number of search rounds. The loop: retrieve -> assess relevance/uncertainty ->
search again if still uncertain (up to a budget) -> synthesize final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

SearchFn = Callable[[str], list[str]]  # query -> retrieved documents
QueryGenFn = Callable[[str, list[str]], tuple[str, float]]  # (goal, docs so far) -> (next query, query uncertainty)
RelevanceFn = Callable[[str, str], float]  # (goal, document) -> relevance/quality score in [0,1]
SynthesizeFn = Callable[[str, list[str]], tuple[str, float]]  # (goal, docs) -> (answer, confidence)


@dataclass
class ResearchTrace:
    queries: list[str] = field(default_factory=list)
    query_uncertainties: list[float] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    rounds: int = 0


class DeepResearchAgent:
    def __init__(
        self,
        search_fn: SearchFn,
        generate_query: QueryGenFn,
        assess_relevance: RelevanceFn,
        synthesize: SynthesizeFn,
        confidence_threshold: float = 0.75,
        uncertainty_threshold: float = 0.4,
        max_rounds: int = 5,
        min_relevance: float = 0.3,
    ):
        self.search_fn = search_fn
        self.generate_query = generate_query
        self.assess_relevance = assess_relevance
        self.synthesize = synthesize
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.max_rounds = max_rounds
        self.min_relevance = min_relevance

    def run(self, goal: str) -> dict:
        trace = ResearchTrace()

        for _ in range(self.max_rounds):
            query, query_uncertainty = self.generate_query(goal, trace.documents)
            trace.queries.append(query)
            trace.query_uncertainties.append(query_uncertainty)
            trace.rounds += 1

            retrieved = self.search_fn(query)
            relevant = [d for d in retrieved if self.assess_relevance(goal, d) >= self.min_relevance]
            trace.documents.extend(relevant)

            answer, confidence = self.synthesize(goal, trace.documents)

            # Stop once we're confident AND the query-generation itself wasn't
            # flagging high uncertainty — per the lecture, uncertainty in the
            # query-generation step is itself a signal, separate from the
            # final answer's confidence.
            if confidence >= self.confidence_threshold and query_uncertainty <= self.uncertainty_threshold:
                return {"answer": answer, "confidence": confidence, "trace": trace, "stopped": "converged"}

        answer, confidence = self.synthesize(goal, trace.documents)
        return {"answer": answer, "confidence": confidence, "trace": trace, "stopped": "max_rounds"}


if __name__ == "__main__":
    # Toy demo: uncertainty and confidence both improve each round as more
    # (fake) documents accumulate.
    def fake_search(query: str) -> list[str]:
        return [f"doc about '{query}'"]

    round_state = {"i": 0}

    def fake_generate_query(goal: str, docs: list[str]) -> tuple[str, float]:
        round_state["i"] += 1
        uncertainty = max(0.0, 0.9 - 0.25 * round_state["i"])
        return f"{goal} detail {round_state['i']}", uncertainty

    def fake_assess_relevance(goal: str, doc: str) -> float:
        return 0.8

    def fake_synthesize(goal: str, docs: list[str]) -> tuple[str, float]:
        confidence = min(0.95, 0.3 + 0.2 * len(docs))
        return f"Answer based on {len(docs)} docs", confidence

    agent = DeepResearchAgent(
        search_fn=fake_search,
        generate_query=fake_generate_query,
        assess_relevance=fake_assess_relevance,
        synthesize=fake_synthesize,
    )
    result = agent.run("What caused the incident?")
    print(result["answer"], "| confidence:", round(result["confidence"], 2), "| stopped:", result["stopped"])
    print("Rounds used:", result["trace"].rounds)
