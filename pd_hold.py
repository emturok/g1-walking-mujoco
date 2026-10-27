import time
import yaml
import numpy as np
import mujoco, mujoco.viewer

# 1. Read the numbers from the yaml file
cfg = yaml.safe_load(open("deploy/deploy_mujoco/configs/g1.yaml"))
xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", ".")
kp = np.array(cfg["kps"])
kd = np.array(cfg["kds"])
q_default = np.array(cfg["default_angles"])

# 2. Load the robot and start with the legs in the default pose
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
d.qpos[7:] = q_default      # the first 7 numbers are the floating base

# 3. Every physics step: compute torques and advance the simulation
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        q = d.qpos[7:]      # current joint angles
        dq = d.qvel[6:]     # current joint speeds
        d.ctrl[:] = kp * (q_default - q) - kd * dq   # the PD formula
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(m.opt.timestep)   # run at real-time speed
        