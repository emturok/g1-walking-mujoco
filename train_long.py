from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_gym_env import G1GymEnv

N_ENVS = 4
TOTAL_TIMESTEPS = 2_000_000  # roughly 250 PPO iterations

vec_env = make_vec_env(G1GymEnv, n_envs=N_ENVS)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tb_logs",
)

# Saves a checkpoint every 50,000 steps, in case training is interrupted
# or you want to check an earlier version later.
checkpoint_callback = CheckpointCallback(
    save_freq=max(50_000 // N_ENVS, 1),
    save_path="./checkpoints",
    name_prefix="g1_ppo",
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save("g1_ppo_final")

print("Training finished.")
