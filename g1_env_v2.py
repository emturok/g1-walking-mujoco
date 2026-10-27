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
        self.dt = self.model.opt.timestep * self.decimation   # 0.02 s per policy step
        self.num_actions = 12

        # --- find the two feet (NEW) ---
        self.foot_ids = [
            i for i in range(self.model.nbody)
            if "ankle_roll" in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
        ]
        assert len(self.foot_ids) == 2, f"expected 2 feet, found {len(self.foot_ids)}"

        # --- joint limits, shrunk to 90% of the real range (NEW) ---
        lo = self.model.jnt_range[1:13, 0]       # joint 0 is the floating base
        hi = self.model.jnt_range[1:13, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        self.q_low = mid - 0.9 * half
        self.q_high = mid + 0.9 * half

        # --- gait settings (NEW) ---
        self.gait_period = 0.8          # seconds for one full left-right cycle
        self.stance_fraction = 0.55     # fraction of the cycle a foot is planted
        self.swing_height = 0.08        # target foot height while swinging [m]
        self.base_height_target = 0.78  # target pelvis height [m]

        # --- reward weights, copied from g1_config.py (NEW) ---
        self.scales = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "base_height": -10.0,
            "dof_acc": -2.5e-7,
            "dof_vel": -1e-3,
            "action_rate": -0.01,
            "dof_pos_limits": -5.0,
            "alive": 0.15,
            "hip_pos": -1.0,
            "contact_no_vel": -0.2,
            "feet_swing_height": -20.0,
            "contact": 0.18,
            "torques": -1e-5,           # inherited from the base config
        }

        # --- episode rules ---
        self.max_steps = 1000           # 20 seconds at 50 Hz
        self.min_height = 0.5
        self.max_tilt_gz = -0.5
        self._clear_memory()

    def _clear_memory(self):
        self.last_action = np.zeros(self.num_actions)
        self.prev_action = np.zeros(self.num_actions)
        self.prev_dq = np.zeros(self.num_actions)
        self.tau = np.zeros(self.num_actions)
        self.step_count = 0
        self.reward_terms = {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.q_default
        mujoco.mj_forward(self.model, self.data)
        self._clear_memory()
        return self._get_obs()

    def step(self, action):
        self.prev_action = self.last_action.copy()     # remember for action_rate
        self.prev_dq = self.data.qvel[6:].copy()       # remember for dof_acc
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
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {"reward_terms": self.reward_terms}

    # ---------- helper functions ----------

    def _rotation(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return R.reshape(3, 3)

    def _phase(self):
        """Where we are in the gait cycle, from 0 to 1. Legs are half a cycle apart."""
        t = self.step_count * self.dt
        phase = (t % self.gait_period) / self.gait_period
        leg_phase = np.array([phase, (phase + 0.5) % 1.0])   # [left, right]
        return phase, leg_phase

    def _feet_contact(self):
        """True for each foot that is touching something."""
        contact = np.zeros(2, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            for k, fid in enumerate(self.foot_ids):
                if fid in bodies:
                    contact[k] = True
        return contact

    def _feet_vel(self):
        """World-frame linear velocity of each foot, shape (2, 3)."""
        v = np.zeros((2, 3))
        res = np.zeros(6)
        for k, fid in enumerate(self.foot_ids):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, fid, res, 0)
            v[k] = res[3:6]               # res = [angular(3), linear(3)]
        return v

    def _get_obs(self):
        d = self.data
        gravity_b = self._rotation().T @ np.array([0.0, 0.0, -1.0])
        phase, _ = self._phase()
        return np.concatenate([
            d.qvel[3:6] * self.ang_vel_scale,                      # 3
            gravity_b,                                             # 3
            self.cmd * self.cmd_scale,                             # 3
            (d.qpos[7:] - self.q_default) * self.dof_pos_scale,    # 12
            d.qvel[6:] * self.dof_vel_scale,                       # 12
            self.last_action,                                      # 12
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],  # 2  (NEW: gait clock)
        ])

    def _get_reward(self):
        d = self.data
        R = self._rotation()
        v_body = R.T @ d.qvel[0:3]
        w_body = d.qvel[3:6]
        gravity_b = R.T @ np.array([0.0, 0.0, -1.0])
        q = d.qpos[7:]
        dq = d.qvel[6:]

        _, leg_phase = self._phase()
        should_stand = leg_phase < self.stance_fraction    # gait schedule, per foot
        contact = self._feet_contact()
        feet_z = d.xpos[self.foot_ids, 2]
        feet_v = self._feet_vel()

        terms = {
            "tracking_lin_vel": np.exp(-np.sum((self.cmd[:2] - v_body[:2]) ** 2) / 0.25),
            "tracking_ang_vel": np.exp(-(self.cmd[2] - w_body[2]) ** 2 / 0.25),
            "lin_vel_z": v_body[2] ** 2,
            "ang_vel_xy": np.sum(w_body[:2] ** 2),
            "orientation": np.sum(gravity_b[:2] ** 2),
            "base_height": (d.qpos[2] - self.base_height_target) ** 2,
            "dof_acc": np.sum(((dq - self.prev_dq) / self.dt) ** 2),
            "dof_vel": np.sum(dq ** 2),
            "action_rate": np.sum((self.last_action - self.prev_action) ** 2),
            "dof_pos_limits": np.sum(np.clip(self.q_low - q, 0, None) + np.clip(q - self.q_high, 0, None)),
            "alive": 1.0,
            "hip_pos": np.sum(q[[1, 2, 7, 8]] ** 2),      # hip roll + hip yaw, both legs
            "contact_no_vel": np.sum(feet_v[contact] ** 2),
            "feet_swing_height": np.sum(((feet_z - self.swing_height) ** 2)[~contact]),
            "contact": np.sum(contact == should_stand),
            "torques": np.sum(self.tau ** 2),
        }
        self.reward_terms = {k: self.scales[k] * v for k, v in terms.items()}
        return max(sum(self.reward_terms.values()), 0.0)   # only positive rewards

    def _is_fallen(self):
        height = self.data.qpos[2]
        gravity_b = self._rotation().T @ np.array([0.0, 0.0, -1.0])
        return bool(height < self.min_height or gravity_b[2] > self.max_tilt_gz)