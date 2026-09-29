# Kinova Gen3 Lite RL Base Code

A minimal starter repository for experimenting with reinforcement learning (RL) on the **Kinova Gen3 Lite** in simulation and, cautiously, on the physical robot.

The goal is to keep the project small and understandable: simulation setup, a Gymnasium-style RL environment, a basic training loop, a real-robot environment adapter, and an example robot control loop.

> **Safety:** Start in simulation. Physical robot examples are templates, not a safety-rated controller. Use a clear workspace, keep people outside the robot's operating area, configure conservative speed/position limits, keep an accessible emergency stop, and supervise every run. Do not send policy outputs directly to hardware until command units, frames, limits, watchdog behavior, and stop conditions have been verified.
> 
## 1. Environment setup

Recommended baseline stack:

- Python 3.10
- MuJoCo for physics simulation
- Gymnasium for the environment interface
- Stable-Baselines3 for PPO (or another compatible algorithm)
- Kinova Kortex API for communication with the physical Gen3 Lite

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install mujoco gymnasium stable-baselines3 numpy
```

Install the Kinova Kortex Python API separately according to the version and installation instructions used by your robot/controller. The hardware adapter should fail closed with a clear message if the API, robot connection, or required configuration is unavailable.

## 2. Simulation setup

1. Place the Gen3 Lite URDF and mesh assets under `assets/robot/` (or use your project's existing model).
2. Convert/prepare the robot model for MuJoCo and verify joint names, axes, limits, base frame, tool frame, and units.
3. Create a scene XML under `assets/scenes/` with a floor, robot, target object/marker, and appropriate collision geometry.
4. Load the scene with MuJoCo and verify that the arm starts in a collision-free configuration.
5. Test reset, stepping, rendering, and joint-limit handling before training.

Minimal model-loading example:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("assets/scenes/scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu)
```

Check that the model's actuators actually correspond to the control interface you intend to use. An actuator control value is not automatically a joint position or velocity unless the MJCF actuator definition makes it so.

## 3. RL simulation environment

Implement a Gymnasium environment with `reset()` and `step()`:

- `reset()` resets the simulator, samples or sets a target, and returns the initial observation.
- `step(action)` validates/clips the action, applies it for a defined number of simulation substeps, computes reward and termination conditions, and returns `(observation, reward, terminated, truncated, info)`.
- Define `observation_space` and `action_space` to exactly match the arrays produced/accepted by the environment.
- Keep simulation timestep, action repeat, joint limits, target distribution, reward terms, and success threshold explicit in configuration.

For a reaching task, a practical starter reward can combine distance-to-target progress, a success bonus, and penalties for excessive action, collisions, or leaving allowed bounds. Document every reward term and evaluate success on held-out target positions.

### Example training loop (PPO)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.sim.env import KinovaReachEnv

env = Monitor(KinovaReachEnv(render_mode=None))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/tensorboard/",
)
model.learn(total_timesteps=500_000)
model.save("checkpoints/kinova_reach_ppo")
env.close()
```

For reproducible experiments, record the random seed, software versions, model asset revision, environment configuration, normalization statistics (if used), and evaluation results. Test the saved policy in a separate evaluation script rather than relying only on training reward.

## 4. Observations and actions: simulation vs. physical robot

An RL policy only receives the observations explicitly exposed by the environment. The simulator may contain more state than the policy is allowed to see. Likewise, hardware feedback depends on the Kortex API, configured feedback sources, and the robot's available sensors.

### Observation choices

| Observation | Simulation | Physical Gen3 Lite |
|---|---|---|
| Joint positions `q` (6 values) | Read directly from MuJoCo joint state | Read actuator/joint feedback through Kortex |
| Joint velocities `q̇` (6) | Read directly from MuJoCo | Read velocity feedback if available/reported |
| End-effector position (3) | Compute with forward kinematics or read site/body pose | Use robot Cartesian feedback or compute from joint feedback and a calibrated model |
| End-effector orientation (e.g. quaternion, 4) | Read/compute from MuJoCo pose | Cartesian feedback or forward kinematics; ensure frame convention matches |
| Target position (3) | Known from task generator | Known if supplied by task/controller; not inherently sensed by robot |
| Relative target vector `target - EE` (3) | Compute from simulated EE and target | Compute from measured/estimated EE and known target |
| Camera/RGB/depth | Simulated camera rendering; may not match real camera | Only if a camera is installed, calibrated, and its data pipeline is implemented |
| Contact/force information | Simulated contacts/forces, subject to model fidelity | Only from supported sensors/feedback; do not assume a wrist force-torque sensor exists |
| Object state | Directly available in simulation if included in observation | Requires perception or external tracking; not automatically available from robot joint feedback |

A useful low-dimensional reaching observation is `[q, q_dot, ee_position, target_position]` (18 values), or `[ee_position, ee_velocity, target_position]` (9 values) if the velocity estimate is available and consistent. These are design options, not a requirement. Normalize units and use the same coordinate frame, ordering, and scaling in simulation and on hardware.

### Action choices

| Action representation | Simulation | Physical robot |
|---|---|---|
| Joint position targets (6) | Set position actuators or track targets with a controller | Send supported joint-angle position commands through Kortex; enforce joint limits and safe speed/acceleration settings |
| Joint velocity commands (6) | Set velocity actuators or use a velocity controller | Use a supported joint-speed command mode; confirm units, limits, command refresh/watchdog requirements |
| Cartesian position target (3) or pose (6/7) | Use inverse kinematics or a Cartesian controller; direct actuator control needs a controller | Use a supported Cartesian action/trajectory interface, with correct base/tool frame and limits |
| Cartesian velocity / twist (typically 6) | Apply through a Cartesian controller or IK velocity mapping | Use supported Cartesian-speed commands; validate frame, units, duration, and stop behavior |
| Torque/effort (6) | Possible only if the MuJoCo model exposes suitable torque actuators | Do not assume direct torque control is exposed/supported on the Gen3 Lite through the installed Kortex interface |

**Important:** An action space is a software contract, not a guarantee that the same command mode exists on both platforms. For sim-to-real work, choose a command representation supported by the physical robot, then implement equivalent semantics in simulation. Clamp actions, rate-limit commands, reject NaN/Inf values, and define timeout and emergency-stop behavior.

## 5. Physical robot environment adapter

Keep hardware I/O separate from the policy and simulator. A hardware environment adapter should:

1. Establish a Kortex session and confirm the expected robot is connected.
2. Read feedback and convert it into the exact observation vector expected by the policy.
3. Validate each policy action and map it to a supported robot command.
4. Send commands at a controlled rate and monitor feedback, timeouts, faults, and workspace limits.
5. Stop motion and release/close resources on normal exit or exceptions.

The real-robot adapter is not automatically interchangeable with a Gymnasium simulator: hardware cannot be reset to arbitrary states, steps take wall-clock time, and failures require explicit recovery. For initial hardware tests, use a supervised control script with fixed, small commands before connecting a learned policy.

## 6. Example supervised control loop

The following is intentionally pseudocode. Replace the adapter calls with the exact Kortex API calls for the installed SDK and verify their signatures, units, command lifetime, and stop semantics against the matching Kinova documentation.

```python
def supervised_control(robot, target_joint_positions, max_steps=100):
    try:
        robot.connect()
        robot.verify_ready()

        for _ in range(max_steps):
            feedback = robot.read_feedback()
            if feedback.has_fault:
                raise RuntimeError(f"Robot fault: {feedback.fault}")

            error = target_joint_positions - feedback.joint_positions
            if max(abs(error)) < 0.02:  # radians; choose a task-appropriate tolerance
                break

            # Adapter must clamp/rate-limit and use a verified Kortex command mode.
            command = robot.make_safe_joint_position_command(target_joint_positions)
            robot.send(command)

            robot.check_workspace_and_timeout()
    finally:
        robot.stop_motion()
        robot.disconnect()
```

Do not treat the pseudocode as executable Kortex API code. Use the official SDK examples for connection/session setup, feedback retrieval, action execution, and stopping.

## 7. Sim-to-real checklist

- [ ] Same joint order, units, coordinate frames, and action scaling in both environments.
- [ ] Verify tool/end-effector frame and forward kinematics against measured poses.
- [ ] Respect physical joint, speed, acceleration, workspace, and payload limits.
- [ ] Include latency, command rate, action hold duration, and timeout behavior in the design.
- [ ] Evaluate on target positions not used during training.
- [ ] Begin with low speed, a clear workspace, an operator at the emergency stop, and a non-learned baseline.
- [ ] Log observations, actions, timestamps, faults, and termination reasons.
- [ ] Never assume simulator contacts, camera observations, or force signals are available on hardware.

## References

- [Kinova Kortex documentation](https://docs.kinovarobotics.com/) — use documentation matching the installed API/firmware version.
- [MuJoCo documentation](https://mujoco.readthedocs.io/)
- [Gymnasium environment API](https://gymnasium.farama.org/)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## License and robot assets

Add a repository license and include only robot descriptions, meshes, and other assets that you are permitted to redistribute. If assets are not redistributable, document where users can obtain them and the expected directory structure.
