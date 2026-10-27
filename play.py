import time
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from g1_env import G1Env

MODEL_PATH = "g1_gait_v4_final"

model = PPO.load(MODEL_PATH)
env = G1Env()
obs = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as v:
    while v.is_running():
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)   # match the training bounds
        obs, reward, terminated, truncated, info = env.step(action)
        v.sync()
        time.sleep(env.model.opt.timestep * env.decimation)
        if terminated or truncated:
            print("Episode ended, resetting.")
            obs = env.reset()