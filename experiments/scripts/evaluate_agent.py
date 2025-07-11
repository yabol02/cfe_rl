import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

import gymnasium as gym
from src.data import DataManager
from src.environments import DiscreteEnv, FlatToStartStepWrapper
from src.utils import load_model
from src.utils import predict_proba, l0_norm, num_subsequences
from stable_baselines3 import DQN

import numpy as np
import torch

np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2)

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")


def calculate_cfe(x1, x2, mask):
    new_mask = np.where(mask, x2, x1)
    return new_mask


exp = "ecg200"
data = DataManager(f"UCR/{exp}", "standard")
env = DiscreteEnv(data, data.model)
env = FlatToStartStepWrapper(env, N=data.get_len(), mode="triangular")


path_model = f"./results/dqn_prueba_{exp}_2.zip"
agent = DQN.load(path_model)

n_episodes = 10
fps = 10
for n in range(n_episodes):
    obs, info = env.reset(train=False)
    orig = obs["original"]
    nun = obs["nun"]
    mask = obs["mask"]
    new = calculate_cfe(orig, nun, mask)
    print(f"{orig=}")
    print(f"{nun=}")
    fig, ax = plt.subplots()
    (lOrig,) = ax.plot(orig[0], c="blue", alpha=0.3, label="Original")
    (lNun,) = ax.plot(nun[0], c="red", alpha=0.3, label="NUN")
    (lNew,) = ax.plot(new[0], c="black", label="CFE")
    ax.legend()
    plt.ion()
    plt.show()
    done, end = False, False
    total_reward = 0
    while not done and not end:
        action, _ = agent.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, end, info = env.step(action)
        total_reward += reward
        print(
            env.get_n_step(),
            action // env.N,
            action % env.N,
            reward,
        )
        new = calculate_cfe(orig, nun, env.get_actual_mask())
        lNew.set_ydata(new[0])
        fig.canvas.draw()
        plt.pause(1 / fps)

    print(f"Episode {n+1}) Total reward = {total_reward}")
    print(f"CFE proba = {predict_proba(data.model, new)[0]}")
    print(f"L0 = {l0_norm(env.get_actual_mask())}")
    print(f"Nº Subsequences = {num_subsequences(env.get_actual_mask())}")
    input("Continue...\n")
    obs, info = env.reset(train=True)
