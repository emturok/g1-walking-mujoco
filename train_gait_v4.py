from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from g1_gym_env import G1GymEnv

N_ENVS = 4
TOTAL_TIMESTEPS = 5_000_000   # fine-tuning needs less than training from scratch


class RewardTermsCallback(BaseCallback):
    """Logs the average of each reward term to TensorBoard."""
    def _on_step(self):
        for info in self.locals["infos"]:
            for name, value in info.get("reward_terms", {}).items():
                self.logger.record_mean(f"reward_terms/{name}", value)
        return True


vec_env = make_vec_env(G1GymEnv, n_envs=N_ENVS)

# v4: start from the trained v2 network instead of a random one
model = PPO.load("g1_gait_final", env=vec_env, tensorboard_log="./tb_logs")

checkpoints = CheckpointCallback(
    save_freq=max(250_000 // N_ENVS, 1),
    save_path="./checkpoints_gait_v4",
    name_prefix="g1_gait_v4",
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([checkpoints, RewardTermsCallback()]),
    tb_log_name="gait_v4",
)
model.save("g1_gait_v4_final")
print("Training finished.")