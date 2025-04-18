import gymnasium as gym
import torch as th
import typing as tp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn import MultiInputPolicy
from ..models import Head1, Head2, MLPExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        input_shape = observation_space.spaces["original"].shape
        input_dims = len(input_shape)

        channels = input_shape[input_dims - 2]
        time_steps = input_shape[input_dims - 1]

        self.cnn_extractor = Head1(channels)
        self.mlp_extractor = Head2(input_dim=3 * channels) # Previously we had output_dim=features_dim

    def forward(self, observations: dict) -> th.Tensor:
        x = th.as_tensor(observations["original"], dtype=th.float32)
        nun = th.as_tensor(observations["nun"], dtype=th.float32)
        mask = th.as_tensor(observations["mask"], dtype=th.float32)

        # 1st head (CNN1D)
        original_features = self.cnn_extractor(x)  # (N, 1, T)
        nun_features = self.cnn_extractor(nun)  # (N, 1, T)

        # Preparing for the 2nd head
        concatenated = th.cat([original_features, nun_features, mask], dim=1)

        # 2nd head(MLP)
        extracted_features = self.mlp_extractor(concatenated)
        return extracted_features


class CustomACPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: tp.Callable[[float], float],
        mask_shape,
        map_function,
        *args,
        **kwargs
    ):
        self.mask_length = mask_shape[-1]
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
        # self.distribution = BernoulliDistribution(mask_shape)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPExtractor(
            self.features_dim,
            mask_length=self.mask_length,
            map_function=self.map_function,
            pi=2,
            vf=1,
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        # actions = distribution.get_actions(deterministic=deterministic)
        action = self.map_function(latent_pi, obs["mask"])
        log_prob = distribution.log_prob(action.to(dtype=th.int))
        action = action.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return action, values, log_prob


class CustomQPolicy(MultiInputPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: tp.Callable[[float], float],
        mask_shape,
        input_dim: int,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=input_dim),
            *args,
            **kwargs,
        )
        self.mask_length = mask_shape
        self.input_dim = input_dim
        self.n_actions = action_space.n

    def _build(self, lr_schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        # print(self.q_net)
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

