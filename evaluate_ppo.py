import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from kinova_env import KinovaEnv

OBS_TYPE = "ee_pos_target"
N_EPISODES = 10
SCENE_XML = r"C:\Users\Zahra Suleymanova\Desktop\KinovaArm\KinovaLift2025\MuJoCo\mujoco_menagerie\kinova_gen3\scene.xml"

# Environment + saved normalization stats
env = DummyVecEnv([lambda: KinovaEnv(SCENE_XML, obs_type=OBS_TYPE)])
env = VecNormalize.load(f"models/{OBS_TYPE}/best_vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

# Trained model
model = PPO.load(f"models/{OBS_TYPE}/best_model.zip", env=env)

rewards, successes, final_dists = [], [], []

for ep in range(N_EPISODES):
    obs, _ = env.env_method("reset_evaluation", episode_index=ep, indices=0)[0]
    done, total_reward, success, dist = False, 0.0, False, 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        done, info = done[0], info[0]

        total_reward += float(reward[0])
        dist = float(info.get("distance", 0.0))
        success = success or info.get("is_success", False)
        env.envs[0].render()

    rewards.append(total_reward)
    successes.append(int(success))
    final_dists.append(dist)
    print(f"Episode {ep + 1}: reward={total_reward:.2f} success={success} final_dist={dist:.3f}")

print(f"\nMean reward:    {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
print(f"Success rate:   {np.mean(successes):.0%}")
print(f"Mean final dist: {np.mean(final_dists):.3f} m")
