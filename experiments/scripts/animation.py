import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from stable_baselines3 import DQN
from src.data import DataManager
from src.environments import DiscreteEnv, FlatToStartStepWrapper
from src.utils import load_model, plot_signal, predict_proba, l0_norm, num_subsequences


def calculate_cfe(x1, x2, mask):
    return np.where(mask, x2, x1)


# --- Setup ---
exp = "ecg200"
data = DataManager(f"UCR/{exp}", "standard")
env = DiscreteEnv(data, data.model)
discrete_env = FlatToStartStepWrapper(env, N=data.get_len(), mode="triangular")
path_model = f"./results/dqn_prueba_{exp}_2.zip"
agent = DQN.load(path_model)

# --- Episode ---
obs, info = discrete_env.reset(train=False)
orig = obs["original"]
nun = obs["nun"]
mask = obs["mask"]
new = calculate_cfe(orig, nun, mask)

steps = []
done, end = False, False
total_reward = 0

while not done and not end:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, end, info = discrete_env.step(int(action))
    total_reward += reward
    steps.append({"step": discrete_env.get_n_step(), "mask": discrete_env.get_actual_mask().copy()})


def init():
    ax.set_title("CFE")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc="upper left", fontsize="x-small")
    ax.grid(True, linestyle="--", alpha=0.7)


def update(frame):
    step_data = steps[frame]
    mask = step_data["mask"]
    plot_signal(orig, nun, mask, ax, dataset=f"Step {step_data['step']}")


plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots()
ani = animation.FuncAnimation(
    fig, update, frames=len(steps), init_func=init, repeat=False, interval=100
)

plt.show()

# --- Final info ---
final_cfe = calculate_cfe(orig, nun, discrete_env.get_actual_mask())
print(f"Total reward = {total_reward}")
print(f"CFE proba = {predict_proba(data.model, final_cfe)[0]}")
print(f"% Changes = {l0_norm(discrete_env.get_actual_mask())/discrete_env.get_actual_mask().size}")
print(f"Nº Subsequences = {num_subsequences(discrete_env.get_actual_mask())}")
