import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=[64, 64])

    def _get_constructor_parameters(self):  # to make the custom policy picklable
        data = super()._get_constructor_parameters()
        data["net_arch"] = self.net_arch
        return data
