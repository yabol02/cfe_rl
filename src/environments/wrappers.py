import gymnasium as gym
import numpy as np


class FlatToStartStepWrapper(gym.ActionWrapper):
    def __init__(self, environment, N, mode="default"):
        super().__init__(environment)
        self.N = N
        self.mode = mode
        self.action_space = gym.spaces.Discrete(self._num_actions(mode))
        self.pairs = self._gen_pairs(mode)

    def action(self, flat_action):
        start, step = self.pairs[int(flat_action)]
        return start, step

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
        return self.env.reset(
            sample=sample,
            nun=nun,
            train=train,
            seed=seed,
            save_res=save_res,
            new_name=new_name,
            options=options,
        )

    def get_cfe(self):
        return self.env.get_cfe()

    def get_n_step(self):
        return self.env.steps

    def get_actual_mask(self):
        return self.env.mask.copy()

    def save_results(self, name):
        return self.env.save_results(name)

    def _num_actions(self, mode):
        if mode == "default":
            return self.N * self.N
        elif mode == "triangular":
            return int(self.N * (self.N + 1) / 2)
        elif mode == "steps":
            return self.N * 4
        elif mode == "value":
            return self.N
        else:
            raise ValueError(f"{mode=} not suported")

    def _gen_pairs(self, mode):
        if mode == "default":
            return self._all_pairs()
        elif mode == "triangular":
            return self._trian_pairs()
        elif mode == "steps":
            return self._steps_pairs()
        elif mode == "value":
            return self._value_pairs()
        else:
            raise ValueError(f"{mode=} not supported")

    def _all_pairs(self):
        flat_actions = np.arange(self.N * self.N)
        start = flat_actions // self.N
        size = flat_actions % self.N
        return np.stack((start, size), axis=1)

    def _trian_pairs(self):
        i_vals, j_vals = np.meshgrid(
            np.arange(self.N), np.arange(self.N), indexing="ij"
        )
        mask = i_vals + j_vals < self.N
        return np.stack((i_vals[mask], j_vals[mask]), axis=-1)

    def _steps_pairs(self):
        steps = [
            1,
            round(self.N * 0.1),
            round(self.N * 0.2),
            round(self.N * 0.4),
        ]
        pairs = [[start, step] for start in range(self.N) for step in steps]
        return np.array(pairs)

    def _value_pairs(self):
        return np.array([[start, 1] for start in range(self.N)])

    def __str__(self):
        return f"<{self.__class__.__name__} {self.env.__class__.__name__}>"

    def __repr__(self):
        return f"<{self.__class__.__name__}(mode={self.mode}, actions={len(self.pairs)}) <{repr(self.env)}>>"


class FlatToStartEndWrapper(gym.ActionWrapper):
    def __init__(self, environment, N):
        super().__init__(environment)
        self.N = N
        self.action_space = gym.spaces.Discrete(N * (N + 1) / 2)
        self.pairs = self._generate_pairs()

    def action(self, flat_action):
        start, end = self.pairs(flat_action)
        return start, end

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
        return self.env.reset(
            sample=sample,
            nun=nun,
            train=train,
            seed=seed,
            save_res=save_res,
            new_name=new_name,
            options=options,
        )

    def _generate_pairs(self):
        start_vals, end_vals = np.triu_indices(self.N)
        return np.stack((start_vals, end_vals), axis=1)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.env.__class__.__name__}>"

    def __repr__(self):
        return f"<{self.__class__.__name__}(mode={self.mode}, actions={len(self.pairs)}) <{repr(self.env)}>>"
