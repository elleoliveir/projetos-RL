import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("FrozenLake-v1", is_slippery=True)

print("Treinando DQN...")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500
)

model.learn(total_timesteps=20000)
model.save("dqn_frozenlake")

env.close()

print("DQN finalizado.")
