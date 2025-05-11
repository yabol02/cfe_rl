import os
import json
import hashlib
import numpy as np
import pandas as pd
from .components import CustomQPolicy, CustomACPolicy
from ..data import DataManager
from ..environments import MyEnv, DiscreteEnv, FlatToStartStepWrapper
from ..utils import (
    load_model,
    predict_proba,
    num_subsequences,
    l0_norm,
    l1_norm,
    l2_norm,
)
from datetime import datetime
from gymnasium import Env, ActionWrapper
from stable_baselines3 import PPO, DQN


def generate_experiment_hash(
    dataset: str, experiment: str, algorithm: str, start_time: str, weights_losses: list
) -> str:
    """
    Generates a stable and unique hash to identify the experiment.

    :param `dataset`: Name of the dataset
    :param `experiment`: Name of the experiment
    :param `algorithm`: Name of the RL algorithm
    :param `start_time`: Timestamp of the experiment start
    :param weight_losses: List with the weights of the losses
    :return: String with the hash of the experiment
    """
    hash_string = (
        f"{dataset}_{experiment}_{algorithm}_{start_time}_{str(weights_losses)}"
    )
    return hashlib.sha256(hash_string.encode()).hexdigest()[:12]


def setup_directories(hash_exp: str, params: dict) -> str:
    """
    Creates the necessary directories and saves the experiment parameters.

    :param `hash_exp`: Unique identifier for the experiment
    :param `params`: Dictionary with the experiment parameters
    :return `results_dir`: Path to the results directory
    """
    results_dir = os.path.join("results", hash_exp)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    return results_dir


def record_experiment_metadata(
    hash_exp: str,
    start_time: str,
    dataset: str,
    algorithm: str,
    super_head: str,
    weights_losses: list,
    mapping_mode: str = None,
) -> None:
    """
    Records the experiment metadata in an Excel file.

    :param `hash_exp`: Unique identifier for the experiment
    :param `start_time`: Start time of the experiment
    :param `dataset`: Name of the dataset
    :param `algorithm`: Name of the RL algorithm
    :param `weight_losses`: List with the weights of the losses
    :param `mapping_mode`: Mode for the wrapper
    """
    excel_path = os.path.join("results", "experiments.xlsx")
    new_entry = {
        "hash": hash_exp,
        "start": start_time,
        "dataset": dataset,
        "algorithm": algorithm,
        "super_head": True if super_head else False,
        "weights_losses": str(weights_losses),
        "mode": mapping_mode,
        "timesteps": 0,
        "total_time": np.nan,
        "reward": np.nan,
        "step": np.nan,
        "proba": np.nan,
        "subsequences": np.nan,
        "num_changes": np.nan,
        "perc_changes": np.nan,
        "L1": np.nan,
        "L2": np.nan,
        "valid": np.nan,
    }

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_excel(excel_path, index=False)


def setup_environment(
    data, model, weights_losses, algorithm: str, mapping_mode: str = None
):
    """
    Sets up the learning environment according to the algorithm and mapping mode.

    :param `data`: DataManager instance with the dataset
    :param `model`: Model instance
    :param `weights_losses`: Weights for the losses
    :param `algorithm`: Name of the RL algorithm
    :param `mapping_mode`: Mode for the FlatToStartStepWrapper
    :return: Environment instance
    """
    if algorithm in ["DQN"]:
        env = DiscreteEnv(data, model, weights_losses)
    else:
        env = MyEnv(data, model, weights_losses)

    if mapping_mode is not None:
        env = FlatToStartStepWrapper(env, N=data.get_len(), mode=mapping_mode)

    return env


def create_dqn_agent(
    env, data: DataManager, hash_exp: str, params: dict, super_head: str = None
):
    """
    Creates a DQN agent with the specified configuration.

    :param `env`: Environment instance
    :param `data`: DataManager instance with the dataset
    :param `hash_exp`: Unique identifier for the experiment
    :param `params`: Dictionary with the experiment parameters
    :param `super_head`: Name of the dataset or None (for super_head configuration)
    :return `agent`: DQN agent instance
    """
    agent = DQN(
        policy=CustomQPolicy,
        env=env,
        policy_kwargs=dict(
            mask_shape=data.get_len(),
            input_dim=data.get_len(),
            super_head=super_head,
            net_arch=[256, 512],
        ),
        learning_rate=params.get("learning_rate", 0.0001),
        buffer_size=params.get("buffer_size", 10_000),
        learning_starts=params.get("learning_starts", 100),
        batch_size=params.get("batch_size", 128),
        tau=params.get("tau", 1),
        gamma=params.get("gamma", 0.99),
        train_freq=params.get("train_freq", 25),
        gradient_steps=params.get("gradient_steps", 1),
        # replay_buffer_class=...,
        # replay_buffer_kwargs=...,
        target_update_interval=params.get("target_update_interval", 10_000),
        exploration_fraction=params.get("exploration_fraction", 0.1),
        exploration_initial_eps=params.get("exploration_initial_eps", 1),
        exploration_final_eps=params.get("exploration_final_eps", 0.05),
        max_grad_norm=params.get("max_grad_norm", 10),
        stats_window_size=params.get("stats_window_size", 100),
        tensorboard_log=f"./results/{hash_exp}",
        verbose=0,
    )
    return agent


def create_ppo_agent(env, data, hash_exp: str, params: dict, super_head: str = None):
    """
    Creates a PPO agent with the specified configuration.

    :param `env`: Environment instance
    :param `data`: DataManager instance with the dataset
    :param `hash_exp`: Unique identifier for the experiment
    :param `params`: Dictionary with the experiment parameters
    :param `super_head`: Name of the dataset or None (for super_head configuration)
    :return `agent`: PPO agent instance
    """
    agent = PPO(
        policy=CustomACPolicy,
        env=env,
        # Add here the parameters for PPO
        verbose=2,
        tensorboard_log=f"./results/{hash_exp}",
    )
    return agent


def build_agent(
    algorithm: str,
    env,
    data,
    params: dict,
    hash_exp: str,
    super_head: str = None,
):
    """
    Builds the learning agent according to the specified algorithm.

    :param `algorithm`: Name of the RL algorithm
    :param `env`: Environment instance
    :param `data`: DataManager instance with the dataset
    :param `params`: Dictionary with the experiment parameters
    :param `hash_exp`: Unique identifier for the experiment
    :param `super_head`: Name of the dataset or None (for super_head configuration)
    :return: Agent instance
    """
    if algorithm == "DQN":
        return create_dqn_agent(env, data, hash_exp, params, super_head)
    elif algorithm == "PPO":
        return create_ppo_agent(env, data, hash_exp, params, super_head)
    else:
        raise ValueError(f"Algorithm '{algorithm}' not supported")


def prepare_experiment(dataset: str, params: dict, dataset_path: str = "./"):
    """
    Main function that prepares all the necessary components for the experiment.

    :param `dataset`: Name of the dataset
    :param `params`: Dictionary with the experiment parameters
    :return: Tuple with (hash_exp, data, env, agent)
    """
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment = params.get("experiment", "fcn")
    scaling = params.get("scaling", "standard")
    algorithm = params.get("algorithm")
    weights_losses = params.get("weights_losses")
    mapping_mode = params.get("mapping_mode")
    super_head = params.get("super_head")
    super_head = dataset if super_head is not None else None
    dataset_path = os.path.join(dataset_path, dataset)
    params["dataset_path"] = dataset_path

    possible_algorithms = ["DQN", "PPO"]
    if algorithm not in possible_algorithms:
        raise ValueError(
            f"The experiment must be some of the next: {possible_algorithms}"
        )

    hash_exp = generate_experiment_hash(
        dataset, experiment, algorithm, start, weights_losses
    )
    setup_directories(hash_exp, params)
    record_experiment_metadata(
        hash_exp, start, dataset, algorithm, super_head, weights_losses, mapping_mode
    )
    model = load_model(dataset=dataset, experiment=experiment)
    data = DataManager(dataset=dataset_path, model=model, scaling=scaling)

    env = setup_environment(data, model, weights_losses, algorithm, mapping_mode)

    agent = build_agent(algorithm, env, data, params, hash_exp, super_head)

    return hash_exp, data, env, agent


def save_agent(hash_experiment: str, data: DataManager, agent, environment):
    agent.save(os.path.join("./results/", hash_experiment, "model.zip"))
    df = pd.read_excel("./results/experiments.xlsx")
    df.loc[df["hash"] == hash_experiment, "timesteps"] = agent.num_timesteps
    start_date = datetime.strptime(
        df[df["hash"] == hash_experiment]["start"].values[0], "%Y-%m-%d %H:%M:%S"
    )
    time_elapsed = datetime.now() - start_date
    time_elapsed = int(time_elapsed.total_seconds())
    df.loc[df["hash"] == hash_experiment, "total_time"] = (
        f"{time_elapsed//60}' {time_elapsed%60}\""
    )
    samples, labels, nuns = data.get_test_samples()
    cfes = obtain_cfes(samples, labels, nuns, environment, agent)
    results = evaluate_cfes(cfes, data.model)
    results.to_excel(f"./results/{hash_experiment}/cfes_info.xlsx", index=False)
    for col in [
        "reward",
        "step",
        "proba",
        "subsequences",
        "num_changes",
        "perc_changes",
        "L1",
        "L2",
        "valid",
    ]:
        if col in results:
            mean = results[col].mean()
            std = results[col].std()
            value = f"{mean:.2f} ± {std:.2f}"
            df.loc[df["hash"] == hash_experiment, col] = value

    df.to_excel("./results/experiments.xlsx", index=False)


def obtain_cfes(samples, labels, nuns, env, agent):
    cfes = list()
    for sample, label, nun in zip(samples, labels, nuns):
        observation, _ = env.reset(sample, nun)
        done, end = False, False
        while not done and not end:
            action, _ = agent.predict(observation, deterministic=True)
            observation, reward, done, end, info = env.step(action)
        cfe_info = env.get_cfe()
        cfe = np.where(cfe_info["mask"], nun, sample)
        cfes.append(
            {
                "sample": sample,
                "nun": nun,
                "cfe": cfe,
                "label": label,
                "mask": cfe_info["mask"],
                "step": cfe_info["step"],
                "reward": cfe_info["reward"],
            }
        )
    return cfes


def evaluate_cfes(cfes, model):
    results = list()
    for data in cfes:
        sample = data["sample"]
        nun = data["nun"]
        cfe = data["cfe"]
        label = data["label"]
        mask = data["mask"]
        proba, pred_class = predict_proba(model, cfe)
        proba = float(proba[0][1 - label])
        subsequences = num_subsequences(mask)
        changes = l0_norm(mask)
        perc_changes = (
            changes / sample.numel() if hasattr(sample, "numel") else sample.size
        )
        l1 = l1_norm(sample, cfe)
        l2 = l2_norm(sample, cfe)
        valid = int(pred_class != label)
        results.append(
            {
                "sample": sample,
                "nun": nun,
                "mask": mask,
                "step": data.get("step"),
                "reward": data.get("reward"),
                "proba": proba,
                "subsequences": subsequences,
                "num_changes": changes,
                "perc_changes": perc_changes,
                "L1": l1,
                "L2": l2,
                "valid": valid,
            }
        )
    df = pd.DataFrame(results)
    return df


def load_saved_experiment(hash_exp: str, model_name: str = "model.zip"):
    """
    Loads a saved agent and its associated environment using existing functions.

    :param `hash_exp`: Unique identifier for the experiment
    :param `model_name`: Name of the model file to load (default: "model.zip")
    :param `dataset_path`: Base path to the dataset directory
    :return `agent`, `env`, `data`: Tuple with the agent, the environment and the DataManager used in the loaded experiment
    """
    results_dir = os.path.join("results", hash_exp)

    params_path = os.path.join(results_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found at {params_path}")

    with open(params_path, "r") as f:
        params = json.load(f)

    dataset_path = params.get("dataset_path")
    dataset = dataset_path.split("/")[-1]
    experiment = params.get("experiment", "fcn")
    algorithm = params.get("algorithm")
    weights_losses = params.get("weights_losses")
    mapping_mode = params.get("mapping_mode")
    scaling = params.get("scaling", "standard")
    super_head = params.get("super_head")
    super_head = dataset if super_head is not None else None

    model_path = os.path.join(results_dir, model_name)
    if not os.path.exists(model_path):
        files = [f for f in os.listdir(results_dir) if f.endswith(".zip")]
        if not files:
            raise FileNotFoundError(f"No saved model found in {results_dir}")
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True
        )
        model_path = os.path.join(results_dir, files[0])
        print(
            f"Model {model_name} not found. Using {os.path.basename(model_path)} instead."
        )

    model = load_model(dataset=dataset, experiment=experiment)
    data = DataManager(dataset=dataset_path, model=model, scaling=scaling)
    env = setup_environment(data, model, weights_losses, algorithm, mapping_mode)

    if algorithm == "DQN":
        agent = DQN.load(
            model_path,
            env=env,
            tensorboard_log=f"./results/{hash_exp}",
        )
    elif algorithm == "PPO":
        agent = PPO.load(model_path, env=env, tensorboard_log=f"./results/{hash_exp}")
    else:
        raise ValueError(f"Algorithm '{algorithm}' not supported")

    print(f"Agent loaded successfully from {model_path}")
    return agent, env, data
