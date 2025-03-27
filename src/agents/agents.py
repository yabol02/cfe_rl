import numpy as np
import typing as tp
import torch as th
from abc import ABC, abstractmethod
from stable_baselines3 import PPO


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


class RandomAgent:
    """
    Agent that makes random decisions on a mask.

    :param `mask_length`: Length of the mask that the agent will modify.
    :param `model`: Classification model for decision making (optional).
    :param `kwargs`: Additional configurations.
    """

    # TODO: Add the classifier to the class constructor (for other agents) => self.model = model
    def __init__(self, mask_length: int, **kwargs):
        self.configuration = {**kwargs}
        self.observation = None
        self.mask = np.ones(mask_length, dtype=np.bool_)

    def decide(self) -> tp.List:
        """
        Makes a random decision on where to start and the size of the transformation

        :return `action`: A list action of the form [start, size]
        """
        # TODO: Add that the observations be passed to the model for decision making => self.model.predict(observation)
        start = int(np.random.uniform(0, 1) * len(self.mask))
        size = int(np.random.uniform(0, 1) * len(self.mask))
        action = [start, size]
        return action

    def map_action(self, action: tp.Tuple[int, int]) -> np.ndarray:
        """
        Transforms the mask according to the given action

        :param `action`: Shape tuple (start of transformation, size of transformation)
        :return: The modified mask
        """
        start, size = action
        start = max(0, start)
        end = min(len(self.mask), start + size)
        self.mask[start:end] = np.logical_not(self.mask[start:end])
        return self.mask

    def step(self, observation):
        """
        Performs a step of the agent with the given observation.

        :param `observation`: Observation from the environment.
        :return `new_mask`: The new modified mask.
        """
        self.observation = observation
        action = (
            self.decide()
        )  # TODO: This mehod must somehow receive the observation from the environment
        print(action)
        new_mask = self.map_action(action)
        return new_mask

    def reset_mask(self):
        """
        Resets the mask to its initial state (all ones)
        """
        return np.ones(len(self.mask))


class PPOAgent(Agent):
    def __init__(self, mask_length, policy, environment, **kwargs):
        super(PPOAgent, self).__init__(mask_length, policy)
        self.observation = None
        self.environment = environment
        self.model = PPO(
            self.policy,
            self.environment,
            policy_kwargs=dict(
                mask_length=mask_length, map_function=map_action_to_mask
            ),
            verbose=0,
        )

    def decide(self):
        action, _ = self.model.predict(self.observation, deterministic=True)
        # print(action)
        start = int(action[0] * len(self.mask))
        end = int(action[1] * len(self.mask))
        return start, end

    def map_action(self, action, mask):
        start, size = action
        start = max(0, start)
        end = min(len(mask), start + size)
        mask[start:end] = np.logical_not(mask[start:end])
        return mask

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
            n_rollout_steps=10,
            # n_episodes=1,
            # n_steps=1,
            # action_noise=None,
            # learning_starts=0,
            # log_interval=1,
            # progress_bar=False,
        )
        self.model.train()


def map_action_to_mask(action, mask):
    start, size = action
    start = th.clamp(start, min=0)
    end = th.clamp(start + size, max=mask.shape[0])

    mask[start:end] ^= 1

    return mask
