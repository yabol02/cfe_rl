import torch as th
from torch import nn
from ..utils import load_model


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
    def __init__(self, input_dim, kernel_size=1):
        super(Head2, self).__init__()
        # self.weights = nn.Parameter(th.ones(3, device="cuda")) # <- Maybe a weighted sum ???
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=kernel_size, padding="same"),
            nn.Flatten(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.network(x)


class SuperHead1(nn.Module):
    def __init__(self, input_dim, dataset, experiment="fcn"):
        super(SuperHead1, self).__init__()
        self.head = load_model(
            dataset, experiment
        ).back_bone  # BE CAREFUL WITH THIS! NOT ALL MODELS WILL HAVE THIS LAYER'S NAME
        for param in self.head.parameters():
            param.requires_grad = False
        self.learnable_layer = nn.Conv1d(
            128, input_dim, kernel_size=1
        )  # AS BEFORE, MAYBE OTHER MODELS DON'T HAVE A 128 DIMENSIONS OUTPUT

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.head(x)
        x = self.learnable_layer(x)
        return x


class DiscreteNetwork(nn.Module):
    def __init__(self, input_dim=64, output_dim=2, sequence_length=24):
        super(DiscreteNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features=sequence_length, out_features=output_dim)
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=sequence_length, out_features=output_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class SharedNetwork(nn.Module):
    def __init__(self, sequence_length, input_dim=64, output_dim=128):
        super(SharedNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, stride=1),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * (sequence_length - 2), output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x.unsqueeze(1))


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
        return self.model(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim=128, output_dim=1):
        super(ValueNetwork, self).__init__()
        self.num = 0
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class MLPExtractor(nn.Module):
    def __init__(self, input_dim, mask_length, shared, pi=2, vf=1):
        super(MLPExtractor, self).__init__()
        self.shared_dim = shared
        self.latent_dim_pi = pi
        self.latent_dim_vf = vf

        self.shared_net = SharedNetwork(
            sequence_length=mask_length,
            input_dim=input_dim,
            output_dim=self.shared_dim,
        )
        self.policy_net = PolicyNetwork(
            input_dim=self.shared_dim, output_dim=self.latent_dim_pi
        )
        self.value_net = ValueNetwork(
            input_dim=self.shared_dim, output_dim=self.latent_dim_vf
        )

        # self.map_function = map_function

    def forward(self, x):
        shared_features = self.shared_net(x)
        policy_output = self.policy_net(shared_features)
        # policy_output = self.map_function(policy_output)
        value_output = self.value_net(shared_features)

        return policy_output, value_output

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        logits = self.policy_net(features)
        new_mask = self.map_function(logits)
        return new_mask

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        shared = self.shared_net(features)
        pred_critic = self.value_net(shared)
        return pred_critic
