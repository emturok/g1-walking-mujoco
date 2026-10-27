from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from g1_gym_env import G1GymEnv

N_ENVS = 4
TOTAL_TIMESTEPS = 10_000_000


class RewardTermsCallback(BaseCallback):
    """Logs the average of each reward term to TensorBoard."""
    def _on_step(self):
        for info in self.locals["infos"]:
            for name, value in info.get("reward_terms", {}).items():
                self.logger.record_mean(f"reward_terms/{name}", value)
        return True


vec_env = make_vec_env(G1GymEnv, n_envs=N_ENVS)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tb_logs",
)

checkpoints = CheckpointCallback(
    save_freq=max(250_000 // N_ENVS, 1),   # about every 250,000 total steps
    save_path="./checkpoints_gait",
    name_prefix="g1_gait",
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoints, RewardTermsCallback()]),
    tb_log_name="gait",
)
model.save("g1_gait_final")
print("Training finished.")
