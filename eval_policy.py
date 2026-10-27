import sys
import numpy as np
from stable_baselines3 import PPO
from g1_env import G1Env

model_path = sys.argv[1]          # the model name typed after the script name
model = PPO.load(model_path)
env = G1Env()

obs = env.reset()
x_start = env.data.qpos[0]        # starting forward position [m]

for t in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, terminated, truncated, info = env.step(np.clip(action, -1.0, 1.0))
    if terminated or truncated:
        break

seconds = (t + 1) * env.dt
distance = env.data.qpos[0] - x_start

print(f"\nModel: {model_path}")
print(f"  survived     : {seconds:.1f} s of 20 ({'fell' if terminated else 'did not fall'})")
print(f"  distance     : {distance:.2f} m forward")
print(f"  average speed: {distance / seconds:.2f} m/s  (command was 0.50)")
