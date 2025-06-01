import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .agents import obtain_cfes
from ..utils import losses


class HyperparamsCallback(BaseCallback):
    def __init__(self, total_timesteps, schedule_config, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.schedule_config = schedule_config

    def _compute_decay(self, initial, final, progress, decay_type="constant"):
        if decay_type == "linear":
            return initial * (1 - progress) + final * progress
        elif decay_type == "exponential":
            return initial * (final / initial) ** progress
        elif decay_type == "logarithmic":
            return final + (initial - final) / (1 * np.log10(1 + 9 * progress))
        elif decay_type == "constant":
            return initial
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")

    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps

        for param_name, config in self.schedule_config.items():
            initial = config["initial"]
            final = config["final"]
            decay_type = config.get("decay", "linear")
            new_value = self._compute_decay(initial, final, progress, decay_type)

            if (
                param_name == "learning_rate"
            ):  # and hasattr(self.model.policy, "optimizer")
                if hasattr(self.model, "policy") and hasattr(
                    self.model.policy, "opitimizer"
                ):
                    for param_group in self.model.policy.optimizer.param_groups:
                        param_group["lr"] = new_value
                self.logger.record("custom/learning", new_value)
            elif (
                param_name == "exploration_rate"
            ):  #  and hasattr(self.model, "exploration_rate")
                self.model.exploration_rate = new_value
                self.logger.record("custom/exploration_rate", new_value)


class LossesCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps,
        tensorboard_path,
        model,
        samples,
        labels,
        nuns,
        env,
        label_nun=0,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_interval = total_timesteps // 10
        self.tb_path = tensorboard_path
        self.predictor = model
        self.samples = samples
        self.labels = labels
        self.nuns = nuns
        self.env = env
        self.num_data = len(samples)
        self.label_nun = label_nun
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_interval:
            self.last_eval = self.num_timesteps
            adv_losses, con_losses, spa_losses, pla_losses = (
                list(),
                list(),
                list(),
                list(),
            )
            cfes = obtain_cfes(
                self.samples, self.labels, self.nuns, self.env, self.model
            )
            for data in cfes:
                cfe = data["cfe"]
                mask = data["mask"]
                sample = data["sample"]
                nun = data["nun"]
                adv_losses.append(
                    losses.adversarial_loss(cfe, self.label_nun, self.predictor, 'cuda')[0]
                )
                con_losses.append(losses.contiguity_loss(mask))
                spa_losses.append(losses.sparsity_loss(mask))
                pla_losses.append(losses.plausability_loss(mask, sample, nun))
            adv_loss = sum(adv_losses) / self.num_data
            con_loss = sum(con_losses) / self.num_data
            spa_loss = sum(spa_losses) / self.num_data
            pla_loss = sum(pla_losses) / self.num_data
            self.logger.record(f"custom/adversarial", adv_loss)
            self.logger.record(f"custom/contiguity", con_loss)
            self.logger.record(f"custom/sparsity", spa_loss)
            self.logger.record(f"custom/plausability", pla_loss)

            if self.verbose > 2:
                print(
                    f"Evaluation at timestep {self.num_timesteps} | Adv: {adv_loss:.4f} | Con: {con_loss:.4f} | Spa: {spa_loss:.4f} | Pla: {pla_loss:.4f}"
                )

        return True
