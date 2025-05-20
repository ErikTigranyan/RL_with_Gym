import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

# 1. Create environment
def make_env():
    return gym.make("Humanoid-v4", render_mode="human")

# 2. Wrap it into a DummyVecEnv (SB3 expects this for VecNormalize)
vec_env = DummyVecEnv([make_env])

# 3. Load VecNormalize statistics
vec_env = VecNormalize.load("path/to/pkl/file", vec_env)

# IMPORTANT: Set to eval mode (disable training stats update)
vec_env.training = False
vec_env.norm_reward = False  # Don't normalize reward at test time

# 4. Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAC.load("path/to/zip/file", device=device)

# 5. Run one episode
obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = vec_env.step(action)
    vec_env.render()

vec_env.close()
