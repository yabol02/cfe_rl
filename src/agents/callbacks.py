import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class HyperparamsCallback(BaseCallback):
    def __init__(self, total_timesteps, schedule_config, verbose = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.schedule_config = schedule_config

    def _compute_decay(self, initial, final, progress, decay_type):
        match decay_type:
            case "linear":
                return initial * (1 - progress) + final * progress
            case "exponential":
                return initial * (final / initial) ** progress
            case "logarithmic":
                return final + (initial - final) / (1 * np.log10(1 + 9 * progress))
            case "constant":
                return initial
            case _:
                raise ValueError(f"Unknown decay type: {decay_type}")
            
    def _on_step(self):
        progress = self.num_timesteps/self.total_timesteps

        for param_name, config in self.schedule_config.items():
            initial = config["initial"]
            final = config["final"]
            decay_type = config.get("decay", "linear")
            new_value = self._compute_decay(initial, final, progress, decay_type)

            if param_name == "learning_rate": # and hasattr(self.model.policy, "optimizer")
                if hasattr(self.model, "policy") and hasattr(self.model.policy, "opitimizer"):
                    for param_group in self.model.policy.optimizer.param_groups:
                        param_group["lr"] = new_value
                self.logger.record("custom/learning", new_value)
            elif param_name == "exploration_rate": #  and hasattr(self.model, "exploration_rate")
                self.model.exploration_rate = new_value
                self.logger.record("custom/exploration_rate", new_value)