import numpy as np
import yaml
import mujoco


class G1Env:
    def __init__(self, cfg_path="deploy/deploy_mujoco/configs/g1.yaml"):
        # --- read the numbers from the yaml file ---
        cfg = yaml.safe_load(open(cfg_path))
        self.kp = np.array(cfg["kps"])
        self.kd = np.array(cfg["kds"])
        self.q_default = np.array(cfg["default_angles"])
        self.action_scale = cfg["action_scale"]
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"])
        self.cmd = np.array(cfg["cmd_init"], dtype=float)
        self.decimation = cfg["control_decimation"]

        # --- load the robot ---
        xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", ".")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.num_actions = 12
        self.last_action = np.zeros(self.num_actions)
        self.tau = np.zeros(self.num_actions)     # most recent motor torques

        # --- episode rules (NEW) ---
        self.max_steps = 1000        # 20 seconds at 50 Hz
        self.min_height = 0.5        # below this height = fallen
        self.max_tilt_gz = -0.5      # gravity z above this = tilted past ~60 degrees
        self.step_count = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.q_default
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(self.num_actions)
        self.tau = np.zeros(self.num_actions)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        self.last_action = np.array(action, dtype=float)
        q_target = self.q_default + self.action_scale * self.last_action
        for _ in range(self.decimation):
            q = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            self.tau = self.kp * (q_target - q) - self.kd * dq
            self.data.ctrl[:] = self.tau
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_fallen()                 # the robot fell
        truncated = self.step_count >= self.max_steps  # time ran out
        return obs, reward, terminated, truncated, {}

    # ---------- helper functions ----------

    def _rotation(self):
        """3x3 matrix describing the body's orientation."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return R.reshape(3, 3)

    def _get_obs(self):
        d = self.data
        gravity_b = self._rotation().T @ np.array([0.0, 0.0, -1.0])
        ang_vel = d.qvel[3:6]
        q = d.qpos[7:]
        dq = d.qvel[6:]

        return np.concatenate([
            ang_vel * self.ang_vel_scale,                # 3
            gravity_b,                                   # 3
            self.cmd * self.cmd_scale,                   # 3
            (q - self.q_default) * self.dof_pos_scale,   # 12
            dq * self.dof_vel_scale,                     # 12
            self.last_action,                            # 12
        ])

    def _get_reward(self):
        d = self.data
        v_body = self._rotation().T @ d.qvel[0:3]   # speed in the robot's own frame
        w_body = d.qvel[3:6]

        lin_err = np.sum((self.cmd[:2] - v_body[:2]) ** 2)
        ang_err = (self.cmd[2] - w_body[2]) ** 2

        r_lin = np.exp(-lin_err / 0.25)              # forward/sideways tracking
        r_ang = np.exp(-ang_err / 0.25)              # turning tracking
        r_alive = 0.15                               # survival bonus
        r_torque = -1e-5 * np.sum(self.tau ** 2)     # penalty for using big torques

        return 1.0 * r_lin + 0.5 * r_ang + r_alive + r_torque

    def _is_fallen(self):
        height = self.data.qpos[2]
        gravity_b = self._rotation().T @ np.array([0.0, 0.0, -1.0])
        return bool(height < self.min_height or gravity_b[2] > self.max_tilt_gz)