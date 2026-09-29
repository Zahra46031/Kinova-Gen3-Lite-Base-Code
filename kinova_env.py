import math
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


class KinovaEnv(gym.Env):
    """Kinova reach task: action = (dx, dy, dz) of the end effector."""

    def __init__(self, model_path, max_steps=300):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.max_steps, self.viewer = max_steps, None
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

    def _obs(self):
        return np.concatenate([self.data.site_xpos[self.site], self.target]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # keyframe 0 = "home"
        offset = np.random.uniform([-0.25, -0.20, -0.25], [0.20, 0.20, 0.25])
        offset *= min(1.0, 0.2 / np.linalg.norm(offset))  # keep target within 0.2 m
        self.target = np.array([0.46, 0.0, 0.43]) + offset
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        action = np.clip(action, -1.0, 1.0)

        # Cartesian delta -> joint delta (Jacobian pseudo-inverse)
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.site)
        self.data.qpos[: self.model.nv] += np.linalg.pinv(jac) @ (action * 0.015)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        to_target = self.target - self.data.site_xpos[self.site]
        dist = float(np.linalg.norm(to_target))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.site)
        vel_toward = float(jac @ self.data.qvel @ to_target / (dist + 1e-6))

        reward = math.exp(-8 * dist) + 0.05 * vel_toward - 1e-3 * float(action @ action)
        reward += 10.0 if dist < 0.03 else 0.0
        info = {"distance": dist, "is_success": dist < 0.03}
        return self._obs(), reward, False, self.t >= self.max_steps, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
