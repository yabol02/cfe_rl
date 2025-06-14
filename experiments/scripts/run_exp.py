import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)


import torch as th
from src import utils
from src.data import DataManager
from src.environments import MyEnv
from src.agents.components import CustomPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN

exp = "chinatown"
model = utils.load_model(exp, "fcn")
data = DataManager(f"UCR/{exp}", model, "standard")


def map_action_to_mask(action: th.Tensor, mask: th.Tensor) -> th.Tensor:
    mask = mask.squeeze(0)
    mask_last_dim = mask.shape[-1]
    for dim in range(len(action)):
        start, size = action[dim]
        start = th.clamp(start * mask_last_dim, min=0).short()
        end = th.clamp((start + size) * mask_last_dim, max=mask_last_dim).short()
        mask[dim, start:end] = th.logical_not(mask[dim, start:end])
    return mask


def lr_schedule(progress):
    return 3e-4 * (1 - progress)


env = MyEnv(data, model, experiment_name=exp)
# check_env(env)
policy_kwargs = dict(mask_shape=data.get_shape(), map_function=map_action_to_mask)
agent = PPO(
    CustomPolicy,  # Policy
    env,  # Environment
    policy_kwargs=policy_kwargs,  # Some arguments for the policy
    learning_rate=lr_schedule,  # Scheduler
    n_steps=2048,  # Steps before updating
    batch_size=64,  # Higher batch size
    n_epochs=10,  # More epochs per update
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # Generalized advantage parameter
    clip_range=0.2,  # Stable clipping
    ent_coef=0.01,  # Entropy regularization
    vf_coef=0.5,  # Weight of the loss of the value function
    verbose=2,
)
agent.learn(total_timesteps=100_000, progress_bar=True)
