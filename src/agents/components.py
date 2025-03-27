import gymnasium as gym
import torch as th
import typing as tp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from ..models import Head1, Head2, MLPExtractor

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        input_shape = observation_space.spaces["original"].shape
        mask_dim = observation_space.spaces["mask"].shape[0]

        channels = input_shape[1]
        time_steps = input_shape[2]

        self.cnn_extractor = Head1(channels)
        self.mlp_extractor = Head2(input_dim=3 * time_steps, output_dim=features_dim)

    def forward(self, observations: dict) -> th.Tensor:
        # TODO: Fix this so you don't have to check every iteration for dimensions
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


class CustomPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: tp.Callable[[float], float],
        mask_length=None,
        map_function=None,
        *args,
        **kwargs
    ):
        self.mask_length = mask_length
        self.map_function = map_function
        kwargs["ortho_init"] = False  # Disabling orthogonal initialization
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPExtractor(
            self.features_dim, self.mask_length, self.map_function
        )