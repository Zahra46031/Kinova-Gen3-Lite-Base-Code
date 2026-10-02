"""Train PPO on KinovaEnv.   python train_ppo.py ee_pos_vel_target --xml assets/scene.xml"""
import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from kinova_env import KinovaEnv, OBS_TYPES

p = argparse.ArgumentParser()
p.add_argument("obs_type", choices=OBS_TYPES)
p.add_argument("--xml", default="assets/scene.xml")
p.add_argument("--timesteps", type=int, default=2_500_000)
args = p.parse_args()

MODEL_DIR, LOG_DIR = f"models/{args.obs_type}", f"logs/{args.obs_type}"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
N_ENVS, N_STEPS = 8, 2048
make_env = lambda: Monitor(KinovaEnv(args.xml, obs_type=args.obs_type))


class SaveVecNormOnBest(BaseCallback):
    """Runs when EvalCallback finds a new best model: saves matching normalization stats."""
    def __init__(self, vecnorm, path): super().__init__(); self.vecnorm, self.path = vecnorm, path
    def _on_step(self): self.vecnorm.save(self.path); return True


if __name__ == "__main__":
    train_env = VecNormalize(DummyVecEnv([make_env] * N_ENVS), norm_obs=True, norm_reward=False)
    eval_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False, training=False)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR,
        eval_freq=10 * N_STEPS,                     # in vec-env steps: ~ every 10 PPO updates
        n_eval_episodes=20, deterministic=True,
        callback_on_new_best=SaveVecNormOnBest(train_env, f"{MODEL_DIR}/best_vecnormalize.pkl"))

    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-4, n_epochs=10,
                n_steps=N_STEPS, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.1, vf_coef=1.0)
    model.learn(args.timesteps, callback=eval_cb)

    model.save(f"{MODEL_DIR}/final_model.zip")
    train_env.save(f"{MODEL_DIR}/final_vecnormalize.pkl")     # policy + its normalization stats: keep together!
    print("Saved to", MODEL_DIR)
