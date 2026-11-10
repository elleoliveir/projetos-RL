import pandas as pd
import matplotlib.pyplot as plt
import os


def carregar_monitor(caminho):
    return pd.read_csv(caminho, comment="#")


def plot_individual(df, nome, saida, window=50):
    df = df.copy()
    df["episode"] = range(1, len(df) + 1)
    df["media_movel"] = df["r"].rolling(window=window).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["media_movel"])
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa média móvel")
    plt.title(f"FrozenLake-v1 – {nome} (janela = {window})")
    plt.grid(True)

    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close()


os.makedirs("graficos", exist_ok=True)

ppo = carregar_monitor("logs/ppo/monitor.csv")
a2c = carregar_monitor("logs/a2c/monitor.csv")
dqn = carregar_monitor("logs/dqn/monitor.csv")

plot_individual(ppo, "PPO", "graficos/ppo_frozenlake.png")
plot_individual(a2c, "A2C", "graficos/a2c_frozenlake.png")
plot_individual(dqn, "DQN", "graficos/dqn_frozenlake.png")

print("Gráficos gerados na pasta graficos/")
