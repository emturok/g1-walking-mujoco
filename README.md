# G1 walking in MuJoCo (WiscoHumanoids new member project)

PPO walking policies for the 12-DoF Unitree G1, trained with stable-baselines3 in MuJoCo on CPU.
See the report PDF for the full method and results.

## Setup
1. Clone https://github.com/unitreerobotics/unitree_rl_gym
2. Copy all files from this repo into the top folder of that clone
3. `pip install mujoco numpy torch gymnasium stable-baselines3 tensorboard pyyaml`

## Run
- Watch a policy: set MODEL_PATH in play.py (e.g. "g1_gait_v4_final"), then `mjpython play.py` (Mac) or `python play.py`
- Measure a policy: `python eval_policy.py g1_gait_v4_final`
- Train: `python train_gait_v4.py` (needs g1_gait_final.zip, the v2 model)

## Files
- g1_env.py: current environment (v4 reward); g1_env_v1/v2/v3.py are earlier versions
- g1_gym_env.py: Gymnasium wrapper for stable-baselines3
- train_*.py: training scripts for each version
- g1_ppo_v1_final.zip, g1_gait_final.zip (v2), g1_gait_v3_final.zip, g1_gait_v4_final.zip: trained models
