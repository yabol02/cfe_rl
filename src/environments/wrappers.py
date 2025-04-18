import gymnasium as gym


class FlatToDivModWrapper(gym.ActionWrapper):
    def __init__(self, environment, N):
        super().__init__(environment)
        self.N = N
        self.action_space = gym.spaces.Discrete(N * N)

    def action(self, flat_action):
        a = flat_action // self.N
        b = flat_action % self.N
        return a, b

    def reset(self, train=True, seed=None, save_res=False, new_name=None, options=None):
        return self.env.reset(
            train=train,
            seed=seed,
            save_res=save_res,
            new_name=new_name,
            options=options,
        )
