from src.agents import (
    prepare_experiment,
    save_agent,
    LossesCallback,
)
from src.utils import load_json_params, generate_param_combinations
import multiprocessing as mp
from itertools import product
from datetime import datetime

DATASETS = ["chinatown", "ecg200"]
DATASETS = ["gunpoint", "beef"]  # , "forda"
params_file = "subspace.json"

parameters = load_json_params(f"./params/agents/{params_file}")
combinations = generate_param_combinations(parameters)


def run_experiment(dataset, combo, lock, exp_idx=None, total_experiments=None):
    import torch

    hash_experiment, data, environment, agent = prepare_experiment(
        dataset=dataset, params=combo, lock=lock, dataset_path="/UCR/", device="cuda"
    )
    
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{exp_idx}/{total_experiments}]" if exp_idx and total_experiments else ""
    print(f"{prefix} Starting experiment {hash_experiment} at {now}...")

    timesteps = combo.get("timesteps", 100_000)
    test_data, test_labels, test_nuns = data.get_test_samples()

    losses_callback = LossesCallback(
        total_timesteps=timesteps,
        tensorboard_path=f"./results/{hash_experiment}",
        model=data.model,
        samples=test_data,
        labels=test_labels,
        nuns=test_nuns,
        env=environment,
    )

    agent.learn(
        total_timesteps=timesteps,
        callback=losses_callback,
        progress_bar=False,
    )

    save_agent(hash_experiment, data, agent, environment, lock)


def run_with_lock(dataset, combo, lock, exp_idx, total_experiments):
    run_experiment(dataset, combo, lock, exp_idx, total_experiments)

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    manager = mp.Manager()
    lock = manager.Lock()

    combos = list(product(DATASETS, combinations))
    total_experiments = len(combos)
    args = [
        (dataset, combo, lock, idx + 1, total_experiments)
        for idx, (dataset, combo) in enumerate(combos)
    ]

    def run_with_lock(dataset, combo, lock, exp_idx, total_experiments):
        run_experiment(dataset, combo, lock, exp_idx, total_experiments)

    with ctx.Pool(processes=2) as pool:
        pool.starmap(run_with_lock, args)