"""
Lecture 6 — PPO -> GRPO -> DAPO, as advantage computation + clipped surrogate
loss. This is the *math*, not a full training loop (no actual model, optimizer,
or backward pass) — plug these into your training step once you have real
per-token log-probs from a model. Uses numpy only.

The point of writing all three side by side is the lineage the lecture draws:
each one is a small, motivated fix on the one before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# PPO: needs a value/critic estimate to compute advantage.
# ---------------------------------------------------------------------------
def ppo_advantage(rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Simplest form: advantage = reward - value estimate. (Real PPO typically
    uses GAE over a trajectory; this is the one-step simplification the
    lecture uses to motivate why GRPO's group baseline is attractive — you
    don't need `values` at all if you have `rewards` for a group.)
    """
    return rewards - values


def clipped_surrogate_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> float:
    """Shared by PPO, GRPO, and DAPO — only what's plugged into `advantages`
    and (for DAPO) whether clip_low == clip_high changes between them.

    clip_low/clip_high being asymmetric (clip_high > clip_low) is exactly
    DAPO's "clip-higher" fix: it lets a token's probability increase further
    than standard symmetric PPO clipping would allow, which the lecture ties
    directly to preventing exploration collapse on already-likely tokens.
    """
    ratio = np.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1 - clip_low, 1 + clip_high) * advantages
    # PPO/GRPO/DAPO all take the *pessimistic* (min) of the two per-token terms.
    per_token_loss = -np.minimum(unclipped, clipped)
    return float(np.mean(per_token_loss))


# ---------------------------------------------------------------------------
# GRPO: no critic. Baseline comes from the group's own reward distribution.
# ---------------------------------------------------------------------------
def grpo_advantage(group_rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """group_rewards: rewards for N completions sampled for the SAME prompt.
    Advantage for each completion is its z-score within the group — this is
    the "use the group for the baseline" move that removed PPO's critic
    network and, per the lecture, is what made scaling RL up to a 32B model
    practical at the course's compute budget.
    """
    mean = group_rewards.mean()
    std = group_rewards.std()
    return (group_rewards - mean) / (std + eps)


# ---------------------------------------------------------------------------
# DAPO: GRPO + three concrete fixes.
# ---------------------------------------------------------------------------
@dataclass
class DAPOConfig:
    clip_low: float = 0.2
    clip_high: float = 0.28  # asymmetric: allow bigger upward moves
    max_response_length: int = 1024  # for overlong filtering
    min_group_reward_std: float = 1e-6  # for dynamic sampling


def dapo_dynamic_sampling_filter(
    groups: list[np.ndarray], config: DAPOConfig
) -> list[np.ndarray]:
    """Drop groups where every sample got (near-)identical reward — the
    zero-advantage case the lecture calls out: no variance means no learning
    signal from that prompt this batch, so don't waste gradient steps on it.
    """
    return [g for g in groups if g.std() > config.min_group_reward_std]


def dapo_overlong_filter(
    responses: list[str], config: DAPOConfig
) -> list[bool]:
    """Mask out degenerate long sequences before they pollute the batch — the
    lecture cites this alone as a 30 -> 36 accuracy jump on their benchmark,
    before any change to the clipping function.
    """
    return [len(r) <= config.max_response_length for r in responses]


def dapo_advantage(group_rewards: np.ndarray, config: DAPOConfig, eps: float = 1e-8) -> np.ndarray:
    """Same as GRPO's advantage; DAPO's actual novelty is upstream (filtering)
    and downstream (asymmetric clipping in the loss) of this step, not the
    advantage formula itself."""
    return grpo_advantage(group_rewards, eps=eps)


def dapo_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    group_rewards: np.ndarray,
    config: DAPOConfig,
) -> float:
    advantages = dapo_advantage(group_rewards, config)
    return clipped_surrogate_loss(
        log_probs, old_log_probs, advantages,
        clip_low=config.clip_low, clip_high=config.clip_high,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # PPO: needs a value estimate alongside rewards.
    rewards = rng.uniform(0, 1, size=8)
    values = rng.uniform(0, 1, size=8)
    print("PPO advantage:", ppo_advantage(rewards, values).round(2))

    # GRPO: 8 completions sampled for ONE prompt, no critic needed.
    group_rewards = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    grpo_adv = grpo_advantage(group_rewards)
    print("GRPO advantage:", grpo_adv.round(2))

    log_probs = rng.uniform(-1, -0.1, size=8)
    old_log_probs = log_probs + rng.normal(0, 0.05, size=8)
    print("Clipped surrogate loss (symmetric, PPO/GRPO style):",
          round(clipped_surrogate_loss(log_probs, old_log_probs, grpo_adv), 4))

    config = DAPOConfig()
    print("Clipped surrogate loss (asymmetric, DAPO style):",
          round(dapo_loss(log_probs, old_log_probs, group_rewards, config), 4))

    # Dynamic sampling filter: drop the degenerate all-same-reward group.
    groups = [group_rewards, np.array([1.0, 1.0, 1.0, 1.0])]
    kept = dapo_dynamic_sampling_filter(groups, config)
    print(f"Kept {len(kept)}/{len(groups)} groups after dynamic sampling filter")
