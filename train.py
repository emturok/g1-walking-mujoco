from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from g1_gym_env import G1GymEnv

N_ENVS = 4  # parallel copies of the robot; kept small since we're on CPU

vec_env = make_vec_env(G1GymEnv, n_envs=N_ENVS)

model = PPO(
    "MlpPolicy",            # a plain feedforward neural network
    vec_env,
    verbose=1,
    tensorboard_log="./tb_logs",
)

# One PPO "iteration" collects n_steps * N_ENVS rollout steps (2048 * 4 by
# default), then runs the training update. 8192 timesteps = exactly one
# iteration, so this checks that collection AND the update both work,
# without waiting for a full run.
model.learn(total_timesteps=8192)
model.save("g1_ppo_smoke_test")

print("Smoke test finished without errors.")
