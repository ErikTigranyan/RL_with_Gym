import gymnasium as gym
import torch
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
import datetime
import os


# === Reward Wrapper: Encourage fast, straight, upright motion ===
class RunningRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        qvel = self.env.unwrapped.data.qvel
        qpos = self.env.unwrapped.data.qpos

        forward_velocity = qvel[0]
        sideways_velocity = abs(qvel[1])
        yaw_rate = abs(qvel[5])
        torso_height = qpos[2]

        forward_bonus = 3.0 * forward_velocity
        lateral_penalty = 2.0 * sideways_velocity
        yaw_penalty = 1.0 * yaw_rate
        upright_bonus = 1.0 * max(0, torso_height - 1.0)

        return reward + forward_bonus - lateral_penalty - yaw_penalty + upright_bonus


# === Early reset if humanoid falls ===
class EarlyResetWrapper(gym.Wrapper):
    def __init__(self, env, min_height=0.8):
        super().__init__(env)
        self.min_height = min_height

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        try:
            height = self.env.unwrapped.data.qpos[2]
            if height < self.min_height:
                done = True
                info["early_reset"] = True
                reward -= 5.0
        except Exception as e:
            print(f"Height check failed: {e}")

        return obs, reward, done, False, info


# === Create parallel environments ===
def make_env(seed):
    def _init():
        env = gym.make("Humanoid-v4")
        env = RunningRewardWrapper(env)
        env = EarlyResetWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


if __name__ == "__main__":
    # === Detect device ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_envs = 8
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # === Load our pretrained model and normalization stats ===
    model_path = "../humanoid_run.zip"
    normalize_path = "../humanoid_runner_vecnormalize_subproc.pkl"

    try:
        model = SAC.load(model_path, env=vec_env)
        print("Loaded pretrained model.")
    except FileNotFoundError:
        print("Error: Pretrained model file not found. Exiting.")
        exit(1)

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // num_envs,
        save_path="./checkpoints",
        name_prefix="sac_humanoid_runner_updated"
    )

    # === Training the model ===
    try:
        model.learn(
            total_timesteps=2_000_000,
            callback=[checkpoint_callback, ProgressBarCallback()]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")

    # === Save the updated model and VecNormalize files with new names in the same directory ===
    updated_model_filename = "./humanoid_run_upd.zip"
    updated_normalize_filename = "./upd_humanoid_runner_vecnormalize_subproc.pkl"

    model.save(updated_model_filename)
    vec_env.save(updated_normalize_filename)
    vec_env.close()

    print(f"Training complete. Model saved to: {updated_model_filename}")
    print(f"VecNormalize stats saved to: {updated_normalize_filename}")

    # === EVALUATION ===
    print("\n=== Starting evaluation ===")

    # Load test env and apply the latest saved normalization
    test_env = SubprocVecEnv([make_env(0)])
    test_env = VecNormalize.load(updated_normalize_filename, test_env)
    test_env.training = False
    test_env.norm_reward = False

    model = SAC.load(updated_model_filename, env=test_env)

    obs = test_env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        time.sleep(0.01)

    test_env.close()
    print("Evaluation complete.")
