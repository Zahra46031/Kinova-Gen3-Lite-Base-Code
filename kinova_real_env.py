"""Gym-style RL environment for the REAL Kinova Gen3 Lite (Kortex API).
Same observations and actions as kinova_env.KinovaEnv (sim). Observations are returned RAW; normalize outside (see deploy.py).
Assumes the robot base frame == sim world frame (true for the setup in this repo: home EE = [0.400, 0.065, 0.428])."""
import math, time
import numpy as np, gymnasium as gym
from gymnasium import spaces
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Base_pb2, Session_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

OBS_DIMS = {"full": 21, "qpos_qvel_diff": 15, "joints_target": 9, "ee_pos_target": 6, "ee_pos_vel_target": 9}
HOME_EE = np.array([0.39980092, 0.06509784, 0.42760288])                                  # sim 'home' keyframe
HOME_QPOS = np.array([0.17063297, -0.94167045, 0.23937329, -0.006403, 0.08143257, -0.06402354])
WS_LO, WS_HI = np.array([0.25, -0.30, 0.20]), np.array([0.65, 0.30, 0.70])               # safety box (m)
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi                                          # wrap angles to [-pi, pi]


class KinovaRealEnv(gym.Env):
    """ACTION: (dx, dy, dz) in [-1, 1] x ee_step metres, sent as a Cartesian velocity (delta / dt) in the base frame."""

    def __init__(self, obs_type="full", ip="192.168.1.10", control_rate=10, ee_step=0.015, max_velocity=0.75,
                 max_steps=300, success_threshold=0.03, vel_alpha=0.3):
        if obs_type not in OBS_DIMS:
            raise NotImplementedError(f"obs_type must be one of {list(OBS_DIMS)} (ee_pose_target needs the EE quaternion)")
        self.obs_type, self.dt, self.ee_step, self.max_velocity = obs_type, 1.0 / control_rate, ee_step, max_velocity
        self.max_steps, self.success_threshold, self.vel_alpha = max_steps, success_threshold, vel_alpha
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIMS[obs_type],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.target, self.t = np.zeros(3, np.float32), 0

        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.transport.connect(ip, 10000)
        info = Session_pb2.CreateSessionInfo(); info.username, info.password = "admin", "admin"
        info.session_inactivity_timeout, info.connection_inactivity_timeout = 60000, 2000
        self.session = SessionManager(self.router); self.session.CreateSession(info)
        self.base = BaseClient(self.router)

    # ---- robot state ----
    def _qpos(self):
        ja = self.base.GetMeasuredJointAngles().joint_angles
        return wrap(np.deg2rad([ja[i].value for i in range(6)])).astype(np.float32)   # Kortex: degrees, 0..360
    def _ee(self):
        p = self.base.GetMeasuredCartesianPose()
        return np.array([p.x, p.y, p.z], np.float32)

    def _velocities(self, q, ee):
        """Kortex gives no velocities through BaseClient: finite differences + low-pass filter."""
        now = time.time()
        if self._prev is not None:
            dt = max(now - self._prev[2], 1e-3); a = self.vel_alpha
            self._qd = a * wrap(q - self._prev[0]) / dt + (1 - a) * self._qd
            self._eev = a * (ee - self._prev[1]) / dt + (1 - a) * self._eev
        self._prev = (q.copy(), ee.copy(), now)
        return self._qd, np.clip(self._eev, -1, 1)          # sim clips ee velocity to +-1

    def _observe(self):
        q, ee = self._qpos(), self._ee()
        qd, eev = self._velocities(q, ee)
        tgt, diff = self.target, ee - self.target             # diff = ee - target (same sign as sim)
        self.distance = float(np.linalg.norm(diff))
        parts = {"full": [q, qd, ee, tgt, diff], "qpos_qvel_diff": [q, qd, diff], "joints_target": [q, tgt],
                 "ee_pos_target": [ee, tgt], "ee_pos_vel_target": [ee, eev, tgt]}[self.obs_type]
        return np.concatenate(parts).astype(np.float32)

    # ---- commands ----
    def _send_velocity(self, v):
        cmd = Base_pb2.TwistCommand(); cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE; cmd.duration = 0
        cmd.twist.linear_x, cmd.twist.linear_y, cmd.twist.linear_z = map(float, v)
        self.base.SendTwistCommand(cmd)

    def stop(self):
        try: self.base.Stop()
        except Exception: pass

    def _move_home(self, timeout=30):
        """Cartesian move to the sim home EE position, keeping the current tool orientation."""
        pose = self.base.GetMeasuredCartesianPose()
        action = Base_pb2.Action(); action.name = "move_to_home"
        tp = action.reach_pose.target_pose
        tp.x, tp.y, tp.z = map(float, HOME_EE); tp.theta_x, tp.theta_y, tp.theta_z = pose.theta_x, pose.theta_y, pose.theta_z
        self.base.ExecuteAction(action)
        start, last, stalled = time.time(), self._ee(), 0
        while time.time() - start < timeout:
            time.sleep(0.5); cur = self._ee()
            if np.linalg.norm(cur - HOME_EE) < 0.02: return
            stalled = stalled + 1 if np.linalg.norm(cur - last) < 0.001 else 0
            if stalled > 5: break                              # stopped moving (~2.5 s)
            last = cur
        if np.linalg.norm(self._ee() - HOME_EE) > 0.05: raise RuntimeError("Could not reach home position")

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not options or "target" not in options: raise ValueError('reset(options={"target": [x, y, z]}) is required')
        self.target = np.asarray(options["target"], np.float32)
        mode = Base_pb2.ServoingModeInformation(); mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        self._move_home()
        self._prev, self._qd, self._eev, self.t = None, np.zeros(6, np.float32), np.zeros(3, np.float32), 0
        for _ in range(5): self._observe(); time.sleep(0.05)   # warm up the velocity filter
        obs = self._observe()
        return obs, {"home_ee_error": float(np.linalg.norm(self._ee() - HOME_EE)),
                     "home_qpos_error": float(np.linalg.norm(self._qpos() - HOME_QPOS)), "distance": self.distance}

    def step(self, action):
        delta = np.clip(np.asarray(action, np.float32), -1, 1) * self.ee_step
        nxt = self._ee() + delta
        if np.any(nxt < WS_LO) or np.any(nxt > WS_HI):         # safety: refuse to leave the workspace box
            self.stop()
            return self._observe(), 0.0, False, True, {"safety_stop": True, "distance": self.distance}
        self._send_velocity(np.clip(delta / self.dt, -self.max_velocity, self.max_velocity))
        time.sleep(self.dt)
        obs = self._observe(); self.t += 1
        success = self.distance < self.success_threshold       # real env ends the episode on success (sim does not)
        truncated = self.t >= self.max_steps
        if success or truncated: self.stop()
        reward = math.exp(-8.0 * self.distance) + (8.0 if success else 0.0)   # distance + success terms of the sim reward
        return obs, reward, success, truncated, {"distance": self.distance, "is_success": success}

    def close(self):
        self.stop()
        try: self.session.CloseSession(); self.transport.disconnect()
        except Exception: pass
