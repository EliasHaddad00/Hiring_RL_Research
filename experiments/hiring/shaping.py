"""Potential-based reward shaping (ablation; PREREGISTRATION.md, "Final run list").

Phi(s) = sum of max(m_i, 0) over the k candidates in the pool with the highest m_i, where k is the
number of open roles and m_i the estimate the agent is shown (all of the pool, shown in a slot or
not). Phi = 0 in every terminal state. Training reward: r + gamma * Phi(s') - Phi(s).
Used in training only; curves, evaluations and reported numbers use the unshaped return.
"""
import gymnasium as gym
import numpy as np


def potential(env) -> float:
    """Phi of the current state of a HiringEnv."""
    if env.done:
        return 0.0
    mu = (env.mu_surv if env.sc.obs_belief == "survival" else env.mu_naive)[env.order]
    return float(np.maximum(np.sort(mu)[::-1][:env.roles_left], 0.0).sum())


class ShapedReward(gym.Wrapper):
    """Goes outside Monitor, so the logged episode return stays unshaped."""

    def __init__(self, env, gamma: float):
        super().__init__(env)
        self.gamma = float(gamma)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.phi = potential(self.env.unwrapped)
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = potential(self.env.unwrapped)
        shaped = reward + self.gamma * phi - self.phi
        self.phi = phi
        return obs, shaped, terminated, truncated, info

    def action_masks(self):
        return self.env.unwrapped.action_masks()
