import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FrozenLake-v1", is_slippery=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=20_000)

model.save("ppo_frozenlake")

env.close()

