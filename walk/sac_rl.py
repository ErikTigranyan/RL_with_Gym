import gymnasium as gym 
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_env():
    def _init():
        env = gym.make("Humanoid-v4", render_mode = "human")
        env = Monitor(env)
        return env
    return _init

def make_eval_env():
    def _init():
        env = gym.make("Humanoid-v4", render_mode = "human")
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        gamma=0.99, 
        batch_size=256, 
        train_freq=1, 
        gradient_steps=1,
        device=device,
        tensorboard_log="./tensorboard_logs/"
    )

    model.learn(total_timesteps=1_000_000)
    model.save("humanoid_sac_subproc")
    env.save("humanoid_vecnormalize_subproc.pkl")

    # Evaluation
    eval_env = SubprocVecEnv([make_eval_env()])
    eval_env = VecNormalize.load("humanoid_vecnormalize_subproc.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done.any():
            obs = eval_env.reset()

    eval_env.close()
