import gymnasium as gym
import torch
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
import datetime
import os

# === Custom Reward Wrapper to Encourage Faster Running and Upright Posture ===
class RunningRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        qvel = self.env.unwrapped.data.qvel
        qpos = self.env.unwrapped.data.qpos

        # Get the forward velocity (x-axis)
        forward_velocity = qvel[0]
        sideways_velocity = abs(qvel[1])
        yaw_rate = abs(qvel[5])
        torso_height = qpos[2]

        # Calculate torso angles (pitch and roll)
        pitch = qpos[7]  # Assuming this is the pitch angle of the torso
        roll = qpos[8]   # Assuming this is the roll angle of the torso

        # Reward for forward speed (increased to make it run faster)
        forward_bonus = 4.0 * forward_velocity

        # Penalize large sideways movement or yaw (drift)
        lateral_penalty = 1.5 * sideways_velocity
        yaw_penalty = 0.8 * yaw_rate

        # Penalize large pitch (forward tilt) or roll (side tilt)
        upright_penalty = 3.0 * (abs(pitch) + abs(roll))  # Penalize large tilts

        # Reward for keeping the torso upright
        upright_bonus = 1.0 * max(0, torso_height - 1.0)

        # Combine all factors into the final reward
        return reward + forward_bonus - lateral_penalty - yaw_penalty - upright_penalty + upright_bonus

# === Joint Limit Penalty Wrapper (Ensures Human-Like Joint Angles) ===
class JointLimitPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, joint_limits=None):
        super().__init__(env)
        self.joint_limits = joint_limits or {
            0: (-1.0, 1.0),  # Example: Joint 0 (hip) should stay within [-1, 1] radians
            1: (-0.5, 0.5),  # Example: Joint 1 (knee) should stay within [-0.5, 0.5] radians
            # Add other joint limits as needed...
        }

    def reward(self, reward):
        qpos = self.env.unwrapped.data.qpos

        # Penalize joint positions that exceed limits
        joint_penalty = 0.0
        for joint_id, (lower, upper) in self.joint_limits.items():
            joint_angle = qpos[joint_id]
            if joint_angle < lower or joint_angle > upper:
                joint_penalty += 5.0  # Add a penalty when joint exceeds limits

        return reward - joint_penalty

# === Arm-Leg Coordination Reward Wrapper (For Human-Like Movement) ===
class GaitRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        qpos = self.env.unwrapped.data.qpos

        # Let's assume the arms and legs have specific joints you can monitor
        # For example, let's take the hip and knee positions
        # Hypothetical joint IDs for arms and legs (customize according to your model)
        left_leg_knee = qpos[10]
        right_leg_knee = qpos[12]
        left_arm_shoulder = qpos[4]
        right_arm_shoulder = qpos[6]

        # Reward for proper arm-leg coordination (basic synchronization example)
        coordination_penalty = abs(left_leg_knee - right_arm_shoulder) + abs(right_leg_knee - left_arm_shoulder)

        return reward - coordination_penalty  # The lower the penalty, the better the coordination

# === Combined Wrapper for Posture, Running, and Coordination ===
class HumanoidRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.running_reward = RunningRewardWrapper(env)
        self.joint_penalty = JointLimitPenaltyWrapper(env)
        self.gait_reward = GaitRewardWrapper(env)

    def reward(self, reward):
        reward = self.running_reward.reward(reward)
        reward = self.joint_penalty.reward(reward)
        reward = self.gait_reward.reward(reward)
        return reward

# === Create Parallel Environments ===
def make_env(seed):
    def _init():
        env = gym.make("Humanoid-v4")
        env = HumanoidRewardWrapper(env)  # Apply our combined custom reward wrapper
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

    # === Load your pretrained model and normalization stats ===
    model_path = "./upd_humanoid_runner_sac_subproc.zip"
    normalize_path = "./upd_humanoid_runner_vecnormalize_subproc.pkl"

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
        name_prefix="2sac_humanoid_runner_updated"
    )

    # === Adjust Learning Rate ===
    learning_rate = 5e-4  # Increased learning rate for faster learning
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="2updated_training_logs/",
        learning_rate=learning_rate,
        device=device,
        ent_coef="auto"  # You can tweak entropy if needed
    )

    # === Training the model ===
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[checkpoint_callback, ProgressBarCallback()]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")

    # === Save the updated model and VecNormalize files with new names in the same directory ===
    updated_model_filename = "./2upd_humanoid_runner_sac_subproc.zip"
    updated_normalize_filename = "./2upd_humanoid_runner_vecnormalize_subproc.pkl"

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
