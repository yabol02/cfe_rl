import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_signal(X, X2, mask, dataset=None):
    mod = X.copy()
    mod[mask] = X2[mask]
    plt.plot(X2, c="k", label="NUN")
    plt.plot(mod, c="r", label="CFE")
    plt.plot(X, c="b", label="Original")

    submasks = extract_submasks(mask)
    for subm in submasks:
        x, y = subm
        plt.axvspan(x - (1 if x > 0 else 0), y, color="red", alpha=0.1)

    plt.legend()
    plt.title(f"CFE{f' - {dataset}' if dataset else ''}")
    plt.show()


def extract_submasks(mask):
    if not mask or not any(mask):
        return []

    submasks = []
    start_idx = None

    for i, value in enumerate(mask):
        if value and start_idx is None:
            start_idx = i
        elif not value and start_idx is not None:
            submasks.append([start_idx, i])
            start_idx = None

    if start_idx is not None:
        submasks.append([start_idx, len(mask)])

    return submasks


dataset = "UCR/chinatown"
X = np.load(f"./data/{dataset}/X_train.npy", allow_pickle=True)
Y = np.load(f"./data/{dataset}/Y_train.npy", allow_pickle=True)
mascara = np.random.choice([True, False], size=X.shape[1]).tolist()

show_data = False
if show_data:
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    indices = np.where(Y == 1)[0]
    for i in indices:
        ax1.plot(X[i], c="r")
    ax1.set_title("Clase 1")

    ax2 = fig.add_subplot(2, 1, 2)
    indices = np.where(Y != 1)[0]
    for i in indices:
        ax2.plot(X[i], c="b")
    ax2.set_title("Clase 2")

    plt.tight_layout().show()


X_samples1 = X[Y == 1]
X_samples2 = X[Y != 1]
X1 = X_samples1[np.random.choice(X_samples1.shape[0], size=1, replace=False)][
    0
].flatten()
X2 = X_samples2[np.random.choice(X_samples2.shape[0], size=1, replace=False)][
    0
].flatten()
plot_signal(X1, X2, mascara, dataset)
