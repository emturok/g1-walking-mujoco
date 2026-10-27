import numpy as np
import gymnasium as gym
from gymnasium import spaces
from g1_env import G1Env


class G1GymEnv(gym.Env):
    """Makes G1Env look like a standard Gymnasium environment,
    which is the shape stable-baselines3 expects."""

    def __init__(self):
        super().__init__()
        self.env = G1Env()

        # The policy's 12 outputs, clipped to [-1, 1].
        # g1_env.py turns these into angles as q_default + 0.25 * action.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # The 47-number observation. Gravity components, velocities, etc.
        # aren't hard-bounded, so we leave the range open.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info