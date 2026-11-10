import gymnasium as gym
from stable_baselines3 import A2C

def main():

    env = gym.make("FrozenLake-v1", is_slippery=True)

    print("Treinando A2C no FrozenLake...")

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1
    )

    model.learn(total_timesteps=50_000)

    model.save("a2c_frozenlake")

    env.close()

    print("Treino finalizado e modelo salvo.")

if __name__ == "__main__":
    main()
