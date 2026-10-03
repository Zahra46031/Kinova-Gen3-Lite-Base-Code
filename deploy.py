"""Example control loop: run a trained PPO policy on the real Gen3 Lite.
   python deploy.py full --target 0.49 0.12 0.40 --ip 192.168.1.10"""
import argparse, pickle, numpy as np
from stable_baselines3 import PPO
from kinova_real_env import KinovaRealEnv

p = argparse.ArgumentParser()
p.add_argument("obs_type", default="full")
p.add_argument("--target", nargs=3, type=float, default=[0.49, 0.12, 0.40], help="robot base frame (m)")
p.add_argument("--which", default="best", choices=["best", "final"])
p.add_argument("--ip", default="192.168.1.10")
p.add_argument("--rate", type=int, default=10, help="control rate (Hz); sim-to-real speed = ee_step * rate")
p.add_argument("--ee_step", type=float, default=0.015, help="training value; use 0.005 for a cautious first run")
args = p.parse_args()

d = f"models/{args.obs_type}"
policy = PPO.load(f"{d}/{'best_model' if args.which == 'best' else 'final_model'}.zip")
with open(f"{d}/{args.which}_vecnormalize.pkl", "rb") as f:
    vecnorm = pickle.load(f)                        # observation stats from training (policy is useless without them)
vecnorm.training = False

env = KinovaRealEnv(obs_type=args.obs_type, ip=args.ip, control_rate=args.rate, ee_step=args.ee_step)
try:
    input("Robot will move to home, then run the policy. Clear the workspace, e-stop in hand. Press Enter...")
    obs, info = env.reset(options={"target": args.target})
    print(f"Home error: EE {info['home_ee_error']:.3f} m, joints {info['home_qpos_error']:.3f} rad")
    for t in range(env.max_steps):
        norm_obs = vecnorm.normalize_obs(obs[None])           # same normalization as training
        if t < 3: print("normalized obs:", np.round(norm_obs[0], 2))   # values far outside +-5 = sim/real mismatch
        action, _ = policy.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        print(f"step {t:3d}  dist={info['distance']:.3f} m")
        if terminated: print("SUCCESS"); break
        if truncated: print("Stopped:", "safety limit" if info.get("safety_stop") else "max steps"); break
except KeyboardInterrupt:
    print("Interrupted")
finally:
    env.close()                                     # always stop the arm
