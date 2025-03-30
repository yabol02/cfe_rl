import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from src import utils
from src.data import DataManager
from src.environments import MyEnv
from src.agents import RandomAgent
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    experiment = "chinatown"
    model = utils.load_model(experiment, "fcn")
    data = DataManager(f"UCR/{experiment}", model, "standard")
    env = MyEnv(data, model, [1 / 3, 1 / 3, 1 / 3], "prueba")
    agent = RandomAgent(data.X_train.shape[2])
    check_env(env)
    obs = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action = agent.step(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(f"{info['step']}: {info['mask']} ==> {reward}")
    print(env.experiment)
    env.reset(save_res=True)
