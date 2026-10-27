import numpy as np
from g1_env import G1Env

env = G1Env()


def run(name, policy):
    obs = env.reset()
    total = 0.0
    for t in range(env.max_steps):
        obs, r, terminated, truncated, _ = env.step(policy(obs))
        total += r
        if t % 10 == 0 or terminated:
            print(f"  step {t:3d}  reward {r:.3f}  height {env.data.qpos[2]:.3f}")
        if terminated or truncated:
            break
    print(f"{name}: ended at step {t + 1} "
          f"(terminated={terminated}, truncated={truncated}), "
          f"total reward {total:.1f}\n")


run("zero action", lambda obs: np.zeros(12))

rng = np.random.default_rng(0)
run("random action", lambda obs: rng.uniform(-1, 1, 12))
