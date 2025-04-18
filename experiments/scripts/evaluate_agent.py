import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

import gymnasium as gym
from src.data import DataManager
from src.environments import DiscreteEnv, FlatToDivModWrapper
from src.utils import load_model
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


exp = "chinatown"
model = load_model(exp, "fcn")
data = DataManager(f"UCR/{exp}", model, "standard")
env = DiscreteEnv(data, model)
discrete_env = FlatToDivModWrapper(env, N=data.get_len())


path_model = f"./results/dqn_prueba_{exp}.zip"
agent = DQN.load(path_model)

n_episodes = 10
fps = 10
for n in range(n_episodes):
    obs, info = discrete_env.reset(train=True)
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
        obs, reward, done, end, info = discrete_env.step(action)
        total_reward += reward
        print(
            discrete_env.env.steps,
            action // discrete_env.N,
            action % discrete_env.N,
            reward,
        )
        new = calculate_cfe(orig, nun, discrete_env.env.mask)
        lNew.set_ydata(new[0])
        fig.canvas.draw()
        plt.pause(1 / fps)

    print(f"Episode {n+1}) Total reward = {total_reward}")
    obs, info = discrete_env.reset(train=True)
    input("Continue...\n")
