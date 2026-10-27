import numpy as np
from g1_env import G1Env

np.set_printoptions(precision=3, suppress=True)

env = G1Env()
obs = env.reset()

print("obs shape :", obs.shape)
print("ang vel   :", obs[0:3])
print("gravity   :", obs[3:6])
print("command   :", obs[6:9])
print("joint pos :", obs[9:21])
print("joint vel :", obs[21:33])
print("last act  :", obs[33:45])

print("\nSending zero actions (hold the default pose):")
for i in range(60):
    obs = env.step(np.zeros(12))
    if i % 10 == 0:
        print(f"step {i:2d}  base height {env.data.qpos[2]:.3f}  gravity {obs[3:6]}")