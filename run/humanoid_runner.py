import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

# === Custom reward wrapper to encourage running ===
class RunningRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        com_velocity = self.env.unwrapped.data.qvel[0]
        bonus = 1.5 * com_velocity
        return reward + bonus

# === Early reset wrapper to detect fall ===
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
                reward -= 5.0  # Penalty for falling
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

    # === Initialize model ===
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log="runner_training_logs/", learning_rate=3e-4)

    # === Load pretrained model ===
    try:
        model = SAC.load("../walk/humanoid_walk.zip", env=vec_env)
        print("Pretrained model loaded.")
    except FileNotFoundError:
        print("No pretrained model found, training from scratch.")

    # === Training callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // num_envs,
        save_path="./checkpoints",
        name_prefix="humanoid_runner"
    )

    # === Start training ===
    try:
        model.learn(
            total_timesteps=2_000_000,
            callback=[checkpoint_callback, ProgressBarCallback()]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving model...")

    # === Save model and normalization stats ===
    model.save("humanoid_runner")
    vec_env.save("humanoid_runner_vecnormalize_subproc.pkl")

    # === Clean shutdown ===
    vec_env.close()
    print("Training finished and environment closed properly.")
