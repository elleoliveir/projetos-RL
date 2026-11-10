import os
import gymnasium as gym

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor


TOTAL_TIMESTEPS = 10000

os.makedirs("logs", exist_ok=True)


def treinar(model_class, nome):

    log_dir = f"logs/{nome}"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = Monitor(env, log_dir)

    if nome == "dqn":
        model = model_class(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500
        )
    else:
        model = model_class(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            gamma=0.99,
            n_steps=2048 if nome == "ppo" else 5
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(f"{nome}_frozenlake")

    env.close()


print("Treinando PPO...")
treinar(PPO, "ppo")

print("Treinando A2C...")
treinar(A2C, "a2c")

print("Treinando DQN...")
treinar(DQN, "dqn")
