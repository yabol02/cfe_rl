import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from src import utils
from src.data import DataManager
from src.environments import DiscreteEnv, FlatToStartStepWrapper
from src.agents.components import CustomACPolicy, CustomQPolicy
from src.agents import evaluate_cfes
from stable_baselines3 import PPO, DQN

algorithm = "DQN"

dataset = "chinatown"
model = utils.load_model(dataset, "fcn")
data = DataManager(os.path.join("UCR", dataset), model, "standard")
env = DiscreteEnv(
    dataset=data, model=model, weights_losses=(1, 1, 1, 0), ones_mask=False
)
w_env = FlatToStartStepWrapper(env, N=data.get_len(), mode="steps")

if algorithm == "PPO":
    policy_kwargs = dict(mask_shape=data.get_shape())
    agent = PPO(
        policy=CustomACPolicy,
        policy_kwargs=policy_kwargs,
        env=w_env,
        gamma=0.25,
        verbose=2,
        seed=0,
    )

elif algorithm == "DQN":
    policy_kwargs = dict(
        mask_shape=data.get_len(),
        input_dim=data.get_len(),
        super_head=None,
        net_arch=[256, 512],
    )
    agent = DQN(
        policy=CustomQPolicy,
        policy_kwargs=policy_kwargs,
        env=w_env,
        gamma=0.25,
        verbose=2,
        seed=0,
    )

agent.learn(
    total_timesteps=100_000,
    # callback=losses_callback,
    progress_bar=True,
)
