import os
import json
import hashlib
import pandas as pd
from .components import CustomQPolicy, CustomACPolicy
from ..data import DataManager
from ..environments import MyEnv, DiscreteEnv, FlatToStartStepWrapper
from ..utils import load_model
from datetime import datetime
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
        "date": start_time,
        "dataset": dataset,
        "algorithm": algorithm,
        "mode": mapping_mode,
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
        verbose=2,
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


def prepare_experiment(dataset: str, params: dict, dataset_path: str = "./data/"):
    """
    Main function that prepares all the necessary components for the experiment.

    :param `dataset`: Name of the dataset
    :param `params`: Dictionary with the experiment parameters
    :return: Tuple with (hash_exp, data, env, agent)
    """
    start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment = params.get("experiment", "fcn")
    scaling = params.get("scaling", "standard")
    algorithm = params.get("algorithm")
    weights_losses = params.get("weights_losses")
    mapping_mode = params.get("mapping_mode")
    super_head = params.get("super_head")
    super_head = dataset if super_head is not None else None

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
        hash_exp, start, dataset, algorithm, weights_losses, mapping_mode
    )
    model = load_model(dataset=dataset, experiment=experiment)
    dataset_path = os.path.join(dataset_path, dataset)
    data = DataManager(dataset=dataset_path, model=model, scaling=scaling)

    env = setup_environment(data, model, weights_losses, algorithm, mapping_mode)

    agent = build_agent(algorithm, env, data, params, hash_exp, super_head)

    return hash_exp, data, env, agent


def load_saved_experiment():
    """This will return environment and agent"""
    raise NotImplementedError()


def evaluate_agent():
    raise NotImplementedError()
