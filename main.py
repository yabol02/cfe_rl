from src.agents import (
    prepare_experiment,
    save_agent,
    LossesCallback,
)
from src.utils import load_json_params, generate_param_combinations
import multiprocessing as mp
from itertools import product
from datetime import datetime
import torch
import gc
import time

DATASETS = ["beef"]  #"chinatown", "ecg200", "gunpoint",
params_file = "exp3_plausibility.json"

parameters = load_json_params(f"./params/agents/{params_file}")
combinations = generate_param_combinations(parameters)


def run_experiment(dataset, combo, lock, exp_idx=None, total_experiments=None):
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

    losses_callback.env = None
    losses_callback.model = None
    losses_callback.samples = None
    losses_callback.labels = None
    losses_callback.nuns = None
    del losses_callback
    del data, environment, agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def worker_process(input_queue):
    while True:
        try:
            args = input_queue.get_nowait()
        except Exception:
            break
        try:
            run_experiment(*args)
        except Exception as e:
            print(f"Error en experimento: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    lock = manager.Lock()

    combos = list(product(DATASETS, combinations))
    total_experiments = len(combos)
    input_queue = manager.Queue()

    for idx, (dataset, combo) in enumerate(combos):
        input_queue.put((dataset, combo, lock, idx + 1, total_experiments))

    num_workers = 2
    processes = []
    for _ in range(num_workers):
        time.sleep(1)
        p = mp.Process(target=worker_process, args=(input_queue,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
