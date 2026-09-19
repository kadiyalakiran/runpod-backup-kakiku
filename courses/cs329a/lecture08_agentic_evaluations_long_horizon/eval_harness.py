"""
Lecture 8 — A minimal long-horizon eval harness, in the spirit of the
METR-style "time horizon" framing: define tasks with a human-time estimate and
a step budget, run an agent against each, and report both raw pass rate AND
the METR-style "longest task duration completed" metric — since the lecture is
explicit that a single aggregate accuracy number hides more than it shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

AgentRunFn = Callable[["Task"], "TaskResult"]


@dataclass
class Task:
    task_id: str
    suite: str  # which task suite this belongs to, e.g. "coding", "research", "ops"
    human_time_estimate_minutes: float
    step_budget: int
    check_success: Callable[[list[dict]], bool]  # trajectory -> did it succeed
    describe_failure: Callable[[list[dict]], str] | None = None


@dataclass
class TaskResult:
    task: Task
    success: bool
    steps_used: int
    trajectory: list[dict] = field(default_factory=list)
    failure_reason: str | None = None


@dataclass
class EvalReport:
    results: list[TaskResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success for r in self.results) / len(self.results)

    def pass_rate_by_suite(self) -> dict[str, float]:
        suites: dict[str, list[TaskResult]] = {}
        for r in self.results:
            suites.setdefault(r.task.suite, []).append(r)
        return {
            suite: sum(r.success for r in rs) / len(rs)
            for suite, rs in suites.items()
        }

    def time_horizon_minutes(self, success_threshold: float = 0.5) -> float:
        """METR-style metric: the longest human-time-estimate for which the
        agent's success rate at or above that duration still clears
        `success_threshold`. This rewards agents that can sustain longer
        tasks, not just ones that ace short ones — the whole point of the
        lecture's critique of single-number accuracy.
        """
        durations = sorted({r.task.human_time_estimate_minutes for r in self.results})
        best = 0.0
        for d in durations:
            at_or_above = [r for r in self.results if r.task.human_time_estimate_minutes >= d]
            if not at_or_above:
                continue
            rate = sum(r.success for r in at_or_above) / len(at_or_above)
            if rate >= success_threshold:
                best = d
        return best

    def failure_modes(self) -> dict[str, int]:
        """Per the lecture's emphasis: look at WHICH categories fail, not
        just the aggregate rate."""
        counts: dict[str, int] = {}
        for r in self.results:
            if not r.success and r.failure_reason:
                counts[r.failure_reason] = counts.get(r.failure_reason, 0) + 1
        return counts


def run_eval(tasks: list[Task], run_agent: AgentRunFn) -> EvalReport:
    results = [run_agent(task) for task in tasks]
    return EvalReport(results=results)


if __name__ == "__main__":
    # Toy demo: a fake agent that succeeds on short tasks and fails more often
    # as the human-time-estimate grows, to exercise the time-horizon metric.
    import random

    def fake_agent(task: Task) -> TaskResult:
        random.seed(hash(task.task_id) % 1000)
        p_success = max(0.05, 1.0 - task.human_time_estimate_minutes / 120)
        success = random.random() < p_success
        reason = None if success else random.choice(["lost_context", "wrong_tool_call", "gave_up_early"])
        return TaskResult(task=task, success=success, steps_used=task.step_budget, failure_reason=reason)

    tasks = [
        Task(f"t{i}", suite="coding" if i % 2 == 0 else "research",
             human_time_estimate_minutes=minutes, step_budget=20,
             check_success=lambda traj: True)
        for i, minutes in enumerate([5, 15, 30, 60, 90, 120, 180])
    ]
    report = run_eval(tasks, fake_agent)
    print("Overall pass rate:", round(report.pass_rate, 2))
    print("By suite:", {k: round(v, 2) for k, v in report.pass_rate_by_suite().items()})
    print("Time horizon (min, @50% success):", report.time_horizon_minutes())
    print("Failure modes:", report.failure_modes())
