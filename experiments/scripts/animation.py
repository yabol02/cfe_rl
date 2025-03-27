import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import json
from src.utils import utils

fig, ax = plt.subplots()
with open("results/prueba.json", "r") as f:
    res = json.load(f)

x = res["sample"]
nun = res["nun"]
steps = res["steps"]

plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots()


def init():
    ax.set_title("CFE")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc="upper left", fontsize="x-small")
    ax.grid(True, linestyle="--", alpha=0.7)


def update(frame):
    step_data = steps[frame]
    mask = np.array(step_data["mask"])
    utils.plot_signal(x, nun, mask, ax, dataset=f"Step {step_data['step']}")


ani = animation.FuncAnimation(
    fig, update, frames=len(steps), init_func=init, repeat=False, interval=100
)

plt.show()
