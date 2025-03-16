import numpy as np
import typing as tp
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract base class for agents.

    :param `mask_length`: Length of the mask that the agent will modify.
    :param `model`: Classification model for decision making (optional).
    :param `kwargs`: Additional configurations.
    """

    def __init__(self, mask_length: int, model=None, **kwargs):
        self.configuration = kwargs
        self.model = model  # Model for decision making
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
    # TODO: Initialize the model/policy using PyTorch or integration with Stable Baselines
    # TODO: Consider adding an action_space and observation_space attributes for better definition
    # TODO: Think of a better way to customize the agent than using kwargs
