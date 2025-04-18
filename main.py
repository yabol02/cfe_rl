import gymnasium as gym
from stable_baselines3 import DQN
from src.agents import CustomQPolicy, CustomFeatureExtractor
from src.data import DataManager
from src.utils import load_model
from src.environments import DiscreteEnv, FlatToDivModWrapper
from stable_baselines3.common.env_checker import check_env

exp = "ecg200"
model = load_model(exp, "fcn")
data = DataManager(f"UCR/{exp}", model, "standard")


env = DiscreteEnv(data, model, [1, 1, 0])
discrete_env = FlatToDivModWrapper(env, N=data.get_len())
agent = DQN(
    CustomQPolicy,
    discrete_env,
    policy_kwargs=dict(
        mask_shape=data.get_len(), input_dim=data.get_len(), net_arch=[256, 512]
    ),
    # tau=0.6,
    target_update_interval=2_000,
    gamma=0,
    train_freq=25,
    buffer_size=10_000,
    learning_starts=5_000,
    batch_size=128,
    exploration_fraction=0.8,
    exploration_final_eps=0.05,
    verbose=2,
    # tensorboard_log="./dqn_logs/",
    # optimize_memory_usage=True,  # útil para buffers grandes <== No funciona con espacio de observaciones de tipo Dict
)
agent.learn(total_timesteps=100_000, log_interval=10, progress_bar=True)
agent.save(f"./results/dqn_prueba_{exp}")