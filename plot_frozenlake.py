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
import pandas as pd
import matplotlib.pyplot as plt
import os


def carregar_monitor(caminho):
    # O arquivo monitor.csv tem comentários nas primeiras linhas (#)
    df = pd.read_csv(caminho, comment="#")
    return df


ppo_path = "logs/ppo/monitor.csv"
a2c_path = "logs/a2c/monitor.csv"
dqn_path = "logs/dqn/monitor.csv"

ppo = carregar_monitor(ppo_path)
a2c = carregar_monitor(a2c_path)
dqn = carregar_monitor(dqn_path)


# Eixo x = episódios
ppo["episode"] = range(1, len(ppo) + 1)
a2c["episode"] = range(1, len(a2c) + 1)
dqn["episode"] = range(1, len(dqn) + 1)


plt.figure(figsize=(10, 6))

plt.plot(ppo["episode"], ppo["r"], label="PPO")
plt.plot(a2c["episode"], a2c["r"], label="A2C")
plt.plot(dqn["episode"], dqn["r"], label="DQN")

plt.xlabel("Episódios")
plt.ylabel("Recompensa por episódio")
plt.title("FrozenLake-v1 – Comparação entre PPO, A2C e DQN")

plt.legend()
plt.grid(True)

os.makedirs("graficos", exist_ok=True)
plt.savefig("graficos/comparacao_frozenlake.png", dpi=150)

plt.show()
