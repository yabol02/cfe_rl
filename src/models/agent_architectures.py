import torch as th
from torch import nn


class Head1(nn.Module):
    def __init__(self, channels):
        super(Head1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=128,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.network(x)


class Head2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Head2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(x.view(1, -1))


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=128, output_dim=2):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x)


class MLPExtractor(nn.Module):
    def __init__(self, input_dim, mask_length=None, map_function=None):
        super(MLPExtractor, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.policy_net = PolicyNetwork(128, 2)
        self.value_net = ValueNetwork(128)

        self.latent_dim_pi = 2
        self.latent_dim_vf = 32

        self.mask_length = mask_length
        self.map_function = map_function

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        shared_features = self.shared_net(x)
        policy_output = self.policy_net(shared_features)
        value_output = self.value_net(shared_features)

        return policy_output, value_output

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        pred_actor = self.policy_net(features)
        if self.map_function is not None:
            print(self.map_function(pred_actor))
        return pred_actor

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        pred_critic = self.value_net(features)
        return pred_critic
