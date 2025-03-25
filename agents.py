import numpy as np
import typing as tp
import torch as th
import gymnasium as gym
from abc import ABC, abstractmethod
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class Agent(ABC):
    """
    Abstract base class for agents.

    :param `mask_length`: Length of the mask that the agent will modify.
    :param `policy`: Classification model for decision making (optional).
    :param `kwargs`: Additional configurations.
    """

    def __init__(self, mask_length: int, policy=None, **kwargs):
        self.configuration = kwargs
        self.policy = policy  # Model for decision making
        self.observation = None
        self.mask = np.ones(mask_length)

    @abstractmethod
    def decide(self) -> tp.List:
        """
        Makes a decision based on the current observation.

        :return `action`: A list action of the form [start, size]
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def map_action(self, action: tp.Tuple[int, int]) -> np.ndarray:
        """
        Transforms the mask according to the given action.

        :param `action`: Shape tuple (start of transformation, size of transformation)
        :return: The modified mask
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def step(self, observation):
        """
        Performs a step of the agent with the given observation.

        :param `observation`: Observation from the environment.
        :return `new_mask`: The new modified mask.
        """
        self.observation = observation
        action = self.decide()
        new_mask = self.map_action(action)
        return new_mask

    def reset_mask(self):
        """
        Resets the mask to its initial state (all ones).
        """
        self.mask = np.ones(len(self.mask))
        return self.mask

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    # TODO: Add methods for saving and loading the agent's state
    # TODO: Implement a method to evaluate the agent's performance
    # TODO: Add compatibility with stable baselines for training and evaluation
    # TODO: Initialize the model/policy using PyTorch or integration with Stable Baselines <-- DONE
    # TODO: Consider adding an action_space and observation_space attributes for better definition
    # TODO: Think of a better way to customize the agent than using kwargs


class Head1(nn.Module):
    def __init__(self, channels):
        super(Head1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=128,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.network(x)


class Head2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Head2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(x.view(1, -1))


class FullFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(FullFeatureExtractor, self).__init__(observation_space, features_dim)

        input_shape = observation_space.spaces["original"].shape
        mask_dim = observation_space.spaces["mask"].shape[0]

        channels = input_shape[1]
        time_steps = input_shape[2]

        self.cnn_extractor = Head1(channels)
        self.mlp_extractor = Head2(input_dim=3 * time_steps, output_dim=features_dim)

    def forward(self, observations: dict) -> th.Tensor:
        x = th.as_tensor(observations["original"], dtype=th.float32)
        x = x.squeeze(0) if x.dim() == 4 else x
        nun = th.as_tensor(observations["nun"], dtype=th.float32)
        nun = nun.squeeze(0) if nun.dim() == 4 else nun
        mask = th.as_tensor(observations["mask"], dtype=th.float32)
        mask = mask.squeeze(0) if mask.dim() == 2 else mask

        # 1st head (CNN1D)
        original_features = self.cnn_extractor(x).squeeze(0)  # (1, 1, Z) -> (1, Z)
        nun_features = self.cnn_extractor(nun).squeeze(0)  # (1, 1, Z) -> (1, Z)

        # Preparing for the 2nd head
        mask_features = mask.unsqueeze(0)  # (Z,) -> (1, Z)
        concatenated = th.cat(
            [original_features, nun_features, mask_features], dim=0
        )  # (3, Z)

        # 2nd head(MLP)
        extracted_features = self.mlp_extractor(concatenated)
        return extracted_features


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=128, output_dim=2):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x)


class MLPExtractor(nn.Module):
    def __init__(self, input_dim):
        super(MLPExtractor, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.policy_net = PolicyNetwork(128, 2)
        self.value_net = ValueNetwork(128)

        self.latent_dim_pi = 2
        self.latent_dim_vf = 32

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        shared_features = self.shared_net(x)
        policy_output = self.policy_net(shared_features)
        value_output = self.value_net(shared_features)

        return policy_output, value_output

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        pred_actor = self.policy_net(features)
        # print(pred_actor)
        return pred_actor

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        pred_critic = self.value_net(features)
        return pred_critic


class PPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: tp.Callable[[float], float],
        *args,
        **kwargs
    ):
        kwargs["ortho_init"] = False  # Disabling orthogonal initialization
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FullFeatureExtractor,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPExtractor(self.features_dim)


class PPOAgent(Agent):
    def __init__(self, mask_length, policy, environment, **kwargs):
        super(PPOAgent, self).__init__(mask_length, policy)
        self.observation = None
        self.environment = environment
        self.model = PPO(self.policy, self.environment, verbose=0)

    def decide(self):
        action, _ = self.model.predict(self.observation, deterministic=True)
        # print(action)
        start = int(action[0] * len(self.mask))
        end = int(action[1] * len(self.mask))
        return start, end

    def map_action(self, action):
        start, size = action
        start = max(0, start)
        end = min(len(self.mask), start + size)
        self.mask[start:end] = np.logical_not(self.mask[start:end])
        return self.mask

    def step(self, observation):
        self.observation = observation
        action = self.decide()
        new_mask = self.map_action(action)
        return new_mask

    def learn(self):
        self.model.collect_rollouts(
            self.environment,
            callback=None,
            rollout_buffer=None,
            n_rollout_steps=10
            # n_episodes=1,
            # n_steps=1,
            # action_noise=None,
            # learning_starts=0,
            # log_interval=1,
            # progress_bar=False,
        )
        self.model.train()
