import gymnasium as gym
import numpy as np
from datetime import datetime
from collections import deque, Counter
from os import makedirs
from json import dump
from ..utils import losses
from ..utils import ArrayTensorEncoder
from ..data import DataManager


class MyEnv(gym.Env):
    def __init__(
        self,
        dataset: DataManager,
        model,
        weights_losses=None,
        experiment_name=None,
        device="cuda",
    ):
        super().__init__()
        self.data = dataset
        self.device = device
        self.model = model.to(self.device).eval()
        self.name = experiment_name
        self.weights = self.compute_weights(weights_losses)
        self.x1 = self.data.get_sample()
        self.x2 = self.data.get_nun(self.x1)
        self.mask = np.ones((self.data.get_dim(), self.data.get_len()), dtype=np.bool_)
        self.steps = 0
        self.last_reward = self.compute_losses(self.x2)
        self.best_reward = self.last_reward.copy()
        self.best_reward_step = 0
        self.best_reward_mask = self.mask.copy()
        self.observation_space = gym.spaces.Dict(
            {
                "original": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.x1.shape, dtype=np.float32
                ),
                "nun": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.x2.shape, dtype=np.float32
                ),
                "mask": gym.spaces.Box(
                    low=0, high=1, shape=self.mask.shape, dtype=np.bool_
                ),
                # "last_reward": gym.spaces.Box(low=-1, high=1, shape=1, dtype=np.float64) # TODO: See how affects passing this to the agent
            }
        )
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=self.mask.shape, dtype=np.bool_
        )
        # self.action_space = gym.spaces.Box(low=0, high=self.data.get_len()-1, shape=(2,), dtype=np.uint32)
        self.actions_buffer = deque(maxlen=16)
        self.experiment = {
            "experiment": self.name,
            "dataset": self.data.name,
            "sample": self.x1,
            "nun": self.x2,
            "datetime_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "datetime_end": None,
            "steps": list(),
        }

    def compute_weights(self, weights):
        """
        Normalizes a list of weights so that they sum to 1. Weight values equal to 0 are allowed and will be preserved in the normalization.

        :param `weights`: List of 4 numerical weights, one for each loss component
        :return `normalized`: Normalized weights whose sum is equal to 1 as a dictionary with where the keys are the different losses ("adversarial", "sparsity", "contiguity" and "plausability")
        :raise `ValueError`: If the list does not contain exactly 4 elements
        :raise `ValueError`: If any weight is less than 0
        :raise `ValueError`: If the sum of all weights is 0
        """
        if not weights:
            return {
                "adversarial": 1 / 4,
                "sparsity": 1 / 4,
                "contiguity": 1 / 4,
                "plausability": 1 / 4,
            }

        if len(weights) != 4:
            raise ValueError("The list must be of size 4, one for each loss.")
        if any(num for num in weights) < 0:
            raise ValueError("All weights must be greater or equal to 0.")
        if sum(weights) == 0:
            raise ValueError("The sum of the weights cannot be equal to 0.")

        normalized_weights = [num / sum(weights) for num in weights]

        return dict(
            zip(
                ["adversarial", "sparsity", "contiguity", "plausability"],
                normalized_weights,
            )
        )

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
        # print(f'{self.steps}) {action=} ==> {self.mask}')
        observation = {"original": self.x1, "nun": self.x2, "mask": self.mask}
        reward = self.reward(new_signal)
        done = self.check_done(action)
        truncated = self.check_end(1000)
        info = self._get_info()
        self.experiment["steps"].append(info.copy())

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

    def compute_losses(self, new_signal) -> float:
        total_reward = 0
        label = self.data.get_predicted_label(self.x2)
        adv, pred = losses.adversarial_loss(new_signal, label, self.model, self.device)
        spa = losses.sparsity_loss(self.mask)
        sub = losses.contiguity_loss(self.mask)
        pla = losses.plausability_loss(self.mask, self.x1, self.x2)
        total_reward += adv * self.weights["adversarial"]
        total_reward += spa * self.weights["sparsity"]
        total_reward += sub * self.weights["contiguity"]
        total_reward += pla * self.weights["plausability"]
        total_reward -= 10 if pred != label else 0
        return total_reward

    def reward(self, new_signal) -> float:
        """
        Calculate reward based on the current state.

        :param `new_signal`: The current modified signal
        :return `reward`: Reward value for the step
        """
        # TODO: Assign a weight to the class penalty
        total_reward = self.compute_losses(new_signal)
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_reward_step = self.steps
            self.best_reward_mask = self.mask.copy()
        reward = total_reward - self.last_reward
        self.last_reward = total_reward
        return reward

    def check_done(self, action) -> bool:
        """
        Checks whether the episode should be terminated based on the repetition pattern of the recent actions.

        Termination is triggered if any of the following conditions are met:
        - The last 5 actions are identical
        - At least 2 distinct actions appear 4 or more times each
        - At least 3 distinct actions appear 3 or more times each

        :param action: The action to be added to the buffer and checked
        :return bool: True if any of the above conditions are satisfied, indicating the episode should end
        """
        self.actions_buffer.append(hash(action))

        if len(self.actions_buffer) >= 5:
            last_actions = list(self.actions_buffer)[-5:]
            if all(a == last_actions[0] for a in last_actions):
                return True

        counts = Counter(self.actions_buffer)

        repeated_4 = sum(1 for count in counts.values() if count >= 4)
        if repeated_4 >= 2:
            return True

        repeated_3 = sum(1 for count in counts.values() if count >= 3)
        if repeated_3 >= 3:
            return True

        return False

    def check_end(self, n: int = 500):
        """
        Verifies wether the episode is terminated by external boundary.

        :param `n`: Number of steps to compute
        :return `bool`: Boolean indicating if the episode must end up now
        """
        return False if self.steps < n else True

    def _get_info(self):
        """
        Obtains the information of the step.

        :return `info`: Experience tuple of the step, wich is of the form => {S_t, A_t, R_t+1, S_t+1} <--- Add more info???
        """
        # TODO: Complete the method
        return {
            "step": self.steps,
            "mask": self.mask.copy(),
            "reward": self.last_reward,
            # "loss": ...,
        }

    def render(self):
        super().render()
        # TODO: Add a method to render an episode
        raise NotImplementedError

    def reset(
        self,
        sample=None,
        nun=None,
        train=True,
        seed=None,
        save_res=False,
        new_name=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.steps = 0
        self.x1 = self.data.get_sample(train=train) if sample is None else sample
        self.x2 = (
            self.data.get_nun(self.x1, train=train, weighted_random=True)
            if nun is None
            else nun
        )
        self.mask = np.ones((self.data.get_dim(), self.data.get_len()), dtype=np.bool_)
        self.last_reward = self.compute_losses(self.x2)
        self.best_reward = self.last_reward.copy()
        self.best_reward_step = 0
        self.best_reward_mask = self.mask.copy()
        self.actions_buffer = deque(maxlen=16)
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
            "dataset": self.data.name,
            "sample": self.x1,
            "nun": self.x2,
            "datetime_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "datetime_end": None,
            "steps": list(),
        }
        return observation, info

    def get_cfe(self):
        return {
            "reward": self.best_reward,
            "step": self.best_reward_step,
            "mask": self.best_reward_mask,
        }

    def save_results(self, def_name=f"exp_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"):
        if self.name is None:
            self.name = def_name
        makedirs("./results", exist_ok=True)
        with open(f"./results/{self.name}.json", "w") as f:
            dump(self.experiment, f, cls=ArrayTensorEncoder, indent=2)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.data.name}>"

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data.name}, model={self.model.__class__.__name__}, weights={self.weights})"


class DiscreteEnv(MyEnv):
    def __init__(
        self,
        dataset,
        model,
        weights_losses=None,
        experiment_name=None,
        device="cuda",
        **kwargs,
    ):
        super().__init__(dataset, model, weights_losses, experiment_name, **kwargs)
        self.action_space = gym.spaces.MultiDiscrete(
            nvec=[self.data.get_len(), self.data.get_len()]
        )  # Other option: start=[0, -self.data.get_len()//2]

    def renew_mask(self, action):
        """
        Updates the mask with the new action.
        """
        mask_last_dim = self.mask.shape[-1]
        start, size = action
        for dim in range(len(self.mask)):
            end = np.clip(start + size, 0, mask_last_dim)
            self.mask[dim, start:end] = np.logical_not(self.mask[dim, start:end])
