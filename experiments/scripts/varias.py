from src import utils
import numpy as np
import torch as th
from src.data import DataManager
from src.agents.components import CustomFeatureExtractor, CustomPolicy


exp = "chinatown"
model = utils.load_model(exp, "fcn")
data = DataManager(f"UCR/{exp}", model, "standard")
x = th.as_tensor(data.get_sample())
nun = th.as_tensor(data.get_nun(x))
mask = th.as_tensor(np.ones(x.shape[2], dtype=np.bool_))

import gymnasium as gym
from src.agents.agents import *

observation_space = gym.spaces.Dict(
    {
        "original": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=x.shape, dtype=np.float32
        ),
        "nun": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=nun.shape, dtype=np.float32
        ),
        "mask": gym.spaces.MultiBinary(n=mask.shape[0]),
    }
)
ext = CustomFeatureExtractor(observation_space)
obs = observation_space.sample()
# featurized = ext.forward(obs)
# print(featurized)

from stable_baselines3 import PPO
from prueba_rl import MyEnv
import src.agents.agents as agents
env = MyEnv(data, model, experiment_name=exp)
my_agent = agents.PPOAgent(x.shape[2], CustomPolicy, env)


action = my_agent.step(obs)
obs, reward, done, truncated, info = env.step(action)
# my_agent.learn()