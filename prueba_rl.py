import utils
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import numpy as np


class RandomAgent:
    """
    :param mask_length: The length of the mask that the agent will modify
    :param kwargs: Additional configurations
    """

    def __init__(self, mask_length, **kwargs):
        # TODO: Add the model to the class constructor => self.model = model
        self.configuration = kwargs
        self.observation = None
        self.mask = np.ones(mask_length)

    def decide(self):
        """
        Makes a random decision on where to start and the size of the transformation
        :param observation:
        :return: A list action [start, size]
        """
        # TODO: Add that the observations be passed to the model for decision making => self.model.predict(observation)
        start = int(np.random.uniform(0, 1) * len(self.mask))
        size = int(np.random.uniform(0, 1) * len(self.mask))
        action = [start, size]
        return action

    def map_action(self, action):
        """
        Transforms the mask according to the given action

        param action: Shape tuple (start of transformation, size of transformation)
        """
        start, size = action
        start = max(0, start)
        end = min(len(self.mask), start + size)
        self.mask[start:end] = np.logical_not(self.mask[start:end])
        return self.mask

    def step(self, observation):
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
        self.mask = np.ones(len(self.mask))


class MyEnv(gym.Env):
    def __init__(self, x1, x2):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.mask = np.ones(self.x1.shape[0], dtype=np.bool_)
        self.steps = 0
        self.last_reward = 0

        self.observation_space = gym.spaces.Dict(
            {
                "original": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.x1.shape, dtype=np.float64
                ),
                "nun": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.x2.shape, dtype=np.float64
                ),
                "mask": gym.spaces.MultiBinary(n=self.mask.shape[0]),
                # "last_reward": gym.spaces.Box(low=-1, high=1, shape=1, dtype=np.float64) # TODO: See how affects pass this
            }
        )

        self.action_space = gym.spaces.MultiBinary(n=self.mask.shape[0])

    def get_sample(self):
        # TODO: By now, the class receives always the same 2 signals. The objective is to pass
        raise NotImplementedError

    def get_nun(self):
        # TODO: By now, the class receives always the same 2 signals. The objective is to pass it 2 datasets and obtain the NUN for each signal.
        raise NotImplementedError

    def step(self, action):
        """
        Executes a simulation step

        :param action: Tuple of the form (beginning of the transformation, size of the transformation)
        :return observation: The updated state of the environment after the action
        :return reward: A scalar value indicating the reward for the current step
        :return done: A boolean indicating if the episode has finished
        :return truncated: A boolean indicating if the episode was cut short
        :return info: A dictionary with additional information
        """
        self.steps += 1
        self.renew_mask(action)
        new_signal = self.compute_cfe()
        observation = {"original": self.x1, "nun": new_signal, "mask": self.mask}

        reward = self.reward(new_signal)

        done = self.check_done()

        truncated = self.check_end(10)

        info = self.get_info()

        return observation, reward, done, truncated, info

    def renew_mask(self, action):
        """
        Updates the mask with the new action
        """
        self.mask = action

    def compute_cfe(self):
        """
        Obtains the new mask by applying the mask

        :return new_signal: The new signal
        """
        return np.where(self.mask == 1, self.x2, self.x1)

    def reward(self, new_signal) -> float:
        """
        Calculate reward based on the current state.

        :param new_signal: The current modified signal
        :return reward: Reward value for the step
        """
        # TODO: Define the full reward functions with the loss metrics in "utils.py"
        total_reward = 0
        total_reward += utils.sparsity_loss(self.mask) * 0.5
        total_reward += utils.contiguity_loss(self.mask) * 0.5
        reward = total_reward - self.last_reward
        self.last_reward = total_reward
        return reward

    def check_done(self) -> bool:
        """
        Verifies wheter the episode ends up naturally

        :return bool: Boolean indicating if the episode has ended or not
        """
        # TODO: Define the stop conditions (I guess we will not stop the training, only if the CFE is very close)
        # An option is something like that: np.allclose(self.x1, self.x2, atol=1e-3)
        return False

    def get_info(self):
        """
        Obtains the information of the step

        :return info: Experience tuple of the step, wich is of the form => {S_t, A_t, R_t+1, S_t+1} <--- Add more info???
        """
        # TODO: Complete the method
        return {
            "step": self.steps,
            "mask": self.mask,
            "loss": ...,
        }

    def check_end(self, n: int = 1000):
        """
        Verifies wether the episode is terminated by external boundary

        :param n: Number of steps to compute
        :return bool: Boolean indicating if the episode must end up now
        """
        return False if self.steps <= n else True

    def reset(self, seed=None):
        # TODO: The method must return 2 new signals, not always the same ones
        super().reset(seed=seed)
        self.steps = 0
        self.last_reward = 0
        self.mask = np.ones(self.x1.shape[0], dtype=np.bool_)
        observation = {"original": self.x1, "nun": self.x2, "mask": self.mask}
        info = self.get_info()
        return observation, info


from time import sleep

if __name__ == "__main__":
    x1 = np.random.rand(10)
    x2 = np.random.rand(10)
    env = MyEnv(x1, x2)
    agent = RandomAgent(x1.shape[0])
    check_env(env)
    obs = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action = agent.step(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(f"{info['step']}: {info['mask']} ==> {reward}")
        sleep(1e-1)
    #     print("Observación:", obs)
    #     print("Recompensa:", reward)
    #     print("Done:", done)
