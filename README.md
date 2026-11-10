# 🧊 Aprendizado por Reforço no FrozenLake: PPO, A2C e DQN

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python) ![Gymnasium](https://img.shields.io/badge/Gymnasium-FrozenLake--v1-0081A5?style=for-the-badge) ![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO_·_A2C_·_DQN-6A4FB8?style=for-the-badge)

Comparação de três algoritmos clássicos de aprendizado por reforço profundo, **PPO**, **A2C** e **DQN**, no ambiente **FrozenLake-v1** do Gymnasium, na versão escorregadia (`is_slippery=True`).

Projeto desenvolvido na disciplina de **Inteligência Computacional** (8º semestre de Engenharia da Computação, UFPA, jan–jul/2026).

---

## 🎯 O problema

No FrozenLake, um agente precisa atravessar um lago congelado (grade 4×4) do ponto de partida até o objetivo sem cair nos buracos. Na versão escorregadia, cada ação só leva para a direção escolhida parte das vezes, então o agente precisa aprender uma política que funcione **apesar da incerteza**. A recompensa é esparsa: 1 ao chegar ao objetivo, 0 em todo o resto.

Isso torna o ambiente um bom teste para comparar como cada algoritmo lida com exploração e recompensa rara.

---

## ⚙️ O que foi feito

- **Treinamento individual** de cada algoritmo com Stable-Baselines3 e política MLP:
  - `treinar_frozenlake.py`: PPO, 20 mil passos
  - `treinar_a2c_frozenlake.py`: A2C, 50 mil passos
  - `treinar_dqn_frozenlake.py`: DQN com replay buffer de 10 mil transições, *batch* de 64, atualização da rede-alvo a cada 500 passos e γ = 0,99
- **Comparação controlada** (`treinar_frozenlake_logs.py`): os três algoritmos treinados com o mesmo número de passos, mesma taxa de aprendizado (1e-3) e mesmo γ, registrando a recompensa de cada episódio com o `Monitor` do Stable-Baselines3.
- **Visualização:**
  - `plot_frozenlake.py`: recompensa por episódio dos três algoritmos no mesmo gráfico
  - `plot_frozenlake_individual.py`: média móvel (janela de 50 episódios) de cada algoritmo separadamente
- **Modelos treinados** salvos em `ppo_frozenlake.zip` e `dqn_frozenlake.zip`, prontos para carregar com `PPO.load()` / `DQN.load()`.

---

## 🧠 Os algoritmos em uma linha

| Algoritmo | Família | Ideia central |
|---|---|---|
| **DQN** | Baseado em valor | Aprende o valor de cada ação com uma rede neural, usando replay buffer e rede-alvo para estabilizar o treino |
| **A2C** | Ator-crítico | Um "ator" escolhe ações e um "crítico" estima o quão boas elas foram, atualizando os dois juntos |
| **PPO** | Ator-crítico | Como o A2C, mas limita o tamanho de cada atualização da política, o que torna o treino mais estável |

---

## 🛠️ Tecnologias

- **Linguagem:** Python
- **Ambiente:** Gymnasium (`FrozenLake-v1`)
- **Algoritmos:** Stable-Baselines3
- **Análise e gráficos:** pandas, Matplotlib

---

## 🚀 Como executar

```bash
pip install gymnasium stable-baselines3 pandas matplotlib

# treino comparativo com registro dos episódios
python treinar_frozenlake_logs.py

# gráficos (salvos em graficos/)
python plot_frozenlake.py
python plot_frozenlake_individual.py
```

---

## 👩‍💻 Autora

**Adryele Oliveira** · [@elleoliveir](https://github.com/elleoliveir) · [Portfólio](https://elleoliveir.github.io)

Desenvolvido para a disciplina de Inteligência Computacional (UFPA, 2026).
