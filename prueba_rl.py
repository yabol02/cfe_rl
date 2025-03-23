import os
import json
import utils
import typing as tp
import numpy as np
import gymnasium as gym
from data_manager import DataManager
from datetime import datetime
from stable_baselines3.common.env_checker import check_env


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


class MyEnv(gym.Env):
    def __init__(
        self, dataset: DataManager, model, weights_losses=None, experiment_name=None
    ):
        super().__init__()
        self.data = dataset
        self.model = model
        self.name = experiment_name
        self.weights = self.compute_weights(weights_losses)
        self.x1 = self.data.get_sample()
        self.x2 = self.data.get_nun(self.x1)
        self.mask = np.ones(self.x1.shape[2], dtype=np.bool_)
        self.steps = 0
        self.last_reward = 0
        self.observation_space = gym.spaces.Dict(
            {
                "original": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.x1.shape, dtype=np.float32
                ),
                "nun": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.x2.shape, dtype=np.float32
                ),
                "mask": gym.spaces.MultiBinary(n=self.mask.shape[0]),
                # "last_reward": gym.spaces.Box(low=-1, high=1, shape=1, dtype=np.float64) # TODO: See how affects passing this to the agent
            }
        )
        self.action_space = gym.spaces.MultiBinary(n=self.mask.shape[0])
        self.experiment = {
            "experiment": self.name,
            "dataset": data.name,
            "sample": self.x1,
            "nun": self.x2,
            "datetime_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "datetime_end": None,
            "steps": list(),
        }

    def compute_weights(self, weights):
        """
        Normalizes a list of weights so that they sum to 1. Weight values equal to 0 are allowed and will be preserved in the normalization.

        :param `weights`: List of 3 numerical weights, one for each loss component
        :return `normalized`: Normalized weights whose sum is equal to 1 as a dictionary with where the keys are the different losses (“adversarial”, “sparsity” and “contiguity”)
        :raise `ValueError`: If the list does not contain exactly 3 elements
        :raise `ValueError`: If any weight is less than 0
        :raise `ValueError`: If the sum of all weights is 0
        """
        if not weights:
            return {"adversarial": 1 / 3, "sparsity": 1 / 3, "contiguity": 1 / 3}

        if len(weights) != 3:
            raise ValueError("The list must be of size 3, one for each loss.")
        if any(num for num in weights) < 0:
            raise ValueError("All weights must be greater or equal to 0.")
        if sum(weights) == 0:
            raise ValueError("The sum of the weights cannot be equal to 0.")

        normalized_weights = [num / sum(weights) for num in weights]

        return dict(zip(["adversarial", "sparsity", "contiguity"], normalized_weights))

    def step(self, action):
        """
        Executes a simulation step.

        :param `action`: Tuple of the form (beginning of the transformation, size of the transformation)
        :return `observation`: The updated state of the environment after the action
        :return `reward`: A scalar value indicating the reward for the current step
        :return `done`: A boolean indicating if the episode has finished
        :return `truncated`: A boolean indicating if the episode was cut short
        :return `info`: A dictionary with additional information
        """
        self.steps += 1
        self.renew_mask(action)
        new_signal = self.compute_cfe()

        observation = {"original": self.x1, "nun": new_signal, "mask": self.mask}
        reward = self.reward(new_signal)
        done = self.check_done()
        truncated = self.check_end(1000)
        info = self._get_info()
        self.experiment["steps"].append(info)

        return observation, reward, done, truncated, info

    def renew_mask(self, action):
        """
        Updates the mask with the new action.
        """
        self.mask = action

    def compute_cfe(self):
        """
        Obtains the new mask by applying the mask.

        :return `new_signal`: The new signal
        """
        return np.where(self.mask == 1, self.x2, self.x1)

    def reward(self, new_signal) -> float:
        """
        Calculate reward based on the current state.

        :param `new_signal`: The current modified signal
        :return `reward`: Reward value for the step
        """
        # TODO: Assign a weight to the class penalty
        total_reward = 0
        adv, pred = utils.adversarial_loss(
            new_signal, self.data.get_predicted_label(self.x1), self.model
        )
        spa = utils.sparsity_loss(self.mask)
        sub = utils.contiguity_loss(self.mask)
        total_reward += adv * self.weights["adversarial"]
        total_reward += spa * self.weights["sparsity"]
        total_reward += sub * self.weights["contiguity"]
        total_reward -= 10 if pred != self.data.get_predicted_label(self.x1) else 0
        reward = total_reward - self.last_reward
        self.last_reward = total_reward
        return reward

    def check_done(self) -> bool:
        """
        Verifies wheter the episode ends up naturally.

        :return `bool`: Boolean indicating if the episode has ended or not
        """
        # TODO: Define the stop conditions (I guess we will not stop the training, only if the CFE is very close)
        # An option is something like that: np.allclose(self.x1, self.x2, atol=1e-3)
        return False

    def _get_info(self):
        """
        Obtains the information of the step.

        :return `info`: Experience tuple of the step, wich is of the form => {S_t, A_t, R_t+1, S_t+1} <--- Add more info???
        """
        # TODO: Complete the method
        return {
            "step": self.steps,
            "mask": self.mask,
            "reward": self.last_reward,
            # "loss": ...,
        }

    def check_end(self, n: int = 1000):
        """
        Verifies wether the episode is terminated by external boundary.

        :param `n`: Number of steps to compute
        :return `bool`: Boolean indicating if the episode must end up now
        """
        return False if self.steps < n else True

    def render(self):
        super().render()
        # TODO: Add a method to render an episode
        raise NotImplementedError

    def reset(self, seed=None, save_res=False, new_name=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_reward = 0
        self.x1 = self.data.get_sample()
        self.x2 = self.data.get_nun(self.x1)
        self.mask = np.ones(self.x1.shape[2], dtype=np.bool_)
        observation = {"original": self.x1, "nun": self.x2, "mask": self.mask}
        info = self._get_info()
        if save_res:
            self.experiment["datetime_end"] = datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S"
            )
            self.save_results()
        if new_name is not None:
            self.name = new_name
        self.experiment = {
            "experiment": self.name,
            "dataset": data.name,
            "sample": self.x1,
            "nun": self.x2,
            "datetime_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "datetime_end": None,
            "steps": list(),
        }
        return observation, info

    def save_results(self):
        if self.name is None:
            self.name = "default_name"
        os.makedirs("./results", exist_ok=True)
        with open(f"./results/{self.name}.json", "w") as f:
            json.dump(self.experiment, f, cls=utils.NumpyArrayEncoder, indent=2)


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
    env.reset(save_res=True)
