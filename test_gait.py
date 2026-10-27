import numpy as np
import mujoco
from g1_env import G1Env

np.set_printoptions(precision=3, suppress=True)

env = G1Env()
obs = env.reset()
print("obs shape :", obs.shape)
print("clock     :", obs[45:47])
print("feet found:", [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in env.foot_ids])

for t in range(40):
    obs, r, terminated, truncated, info = env.step(np.zeros(12))
    if t % 10 == 0:
        print(f"\nstep {t}  total reward {r:.3f}  "
              f"contact {env._feet_contact()}  feet z {env.data.xpos[env.foot_ids, 2]}")
        for name, value in info["reward_terms"].items():
            print(f"    {name:18s} {value:9.4f}")
    if terminated:
        print(f"\nfell at step {t}")
        break