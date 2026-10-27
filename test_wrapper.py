from stable_baselines3.common.env_checker import check_env
from g1_gym_env import G1GymEnv

env = G1GymEnv()
check_env(env)
print("check_env passed with no errors")