"""
Lecture 4 — ReAct: interleaved Thought -> Action -> Observation.

Minimal loop: at each step the model produces a thought and an action (a tool
call), the tool executes, the observation feeds back into the next prompt. The
transcript's two named failure modes are worth testing against once this is
wired to a real model: (1) hallucinated actions/observations when not fine-tuned,
(2) cascading errors on long trajectories — an early bad observation poisons
everything after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The model produces one of these each step, parsed from its raw output.
@dataclass
class Step:
    thought: str
    action: str  # tool name
    action_input: str


ParseStepFn = Callable[[str], Step]
LLMCallFn = Callable[[str], str]  # prompt -> raw model output for one step
ToolFn = Callable[[str], str]  # action_input -> observation string


@dataclass
class ReActAgent:
    llm_call: LLMCallFn
    parse_step: ParseStepFn
    tools: dict[str, ToolFn]
    max_steps: int = 8
    trajectory: list[dict] = field(default_factory=list)

    def build_prompt(self, goal: str) -> str:
        history = "\n".join(
            f"Thought: {t['thought']}\nAction: {t['action']}[{t['action_input']}]\n"
            f"Observation: {t['observation']}"
            for t in self.trajectory
        )
        tool_names = ", ".join(self.tools.keys())
        return (
            f"Goal: {goal}\n"
            f"Available tools: {tool_names}\n"
            f"{history}\n"
            "Thought:"
        )

    def run(self, goal: str) -> dict:
        """Returns the final observation/answer plus the full trajectory, so
        you can inspect exactly where (if anywhere) it went off the rails —
        that trajectory log is what fine-tuning on successful runs, per the
        lecture, actually consumes."""
        for step_num in range(self.max_steps):
            prompt = self.build_prompt(goal)
            raw_output = self.llm_call(prompt)
            step = self.parse_step(raw_output)

            if step.action == "finish":
                self.trajectory.append(
                    {**step.__dict__, "observation": "(done)"}
                )
                return {"answer": step.action_input, "trajectory": self.trajectory, "steps": step_num + 1}

            tool = self.tools.get(step.action)
            observation = (
                tool(step.action_input)
                if tool is not None
                else f"ERROR: unknown tool '{step.action}'"
            )
            self.trajectory.append({**step.__dict__, "observation": observation})

        return {"answer": None, "trajectory": self.trajectory, "steps": self.max_steps, "error": "max_steps_exceeded"}


if __name__ == "__main__":
    # Toy demo: a "calculator" tool and a scripted fake LLM that solves a
    # two-step arithmetic goal via ReAct.
    scripted_outputs = [
        "I should compute 6*7 first. Action: calculator[6*7]",
        "That's 42, now add 8. Action: calculator[42+8]",
        "That's 50, I'm done. Action: finish[50]",
    ]
    call_count = {"i": 0}

    def fake_llm_call(prompt: str) -> str:
        out = scripted_outputs[call_count["i"]]
        call_count["i"] += 1
        return out

    def fake_parse_step(raw: str) -> Step:
        thought, rest = raw.split("Action:", 1)
        action, action_input = rest.strip().split("[", 1)
        return Step(thought=thought.strip(), action=action.strip(), action_input=action_input.rstrip("]"))

    def calculator(expr: str) -> str:
        return str(eval(expr))  # toy only — never eval untrusted input for real

    agent = ReActAgent(
        llm_call=fake_llm_call,
        parse_step=fake_parse_step,
        tools={"calculator": calculator},
    )
    result = agent.run("What is 6*7, then add 8?")
    print(result["answer"])
    for t in result["trajectory"]:
        print(t)
