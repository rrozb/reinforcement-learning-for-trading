from typing import Tuple

import torch.nn as nn
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from torch.distributions import Categorical


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


class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU,
                 *args, **kwargs):
        super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                               net_arch=net_arch, activation_fn=activation_fn,
                                               *args, **kwargs)
        # Assuming observation space is Box type and action space is discrete
        input_dim = observation_space.shape[-1]
        n_actions = action_space.n

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, 64)  # You can change 64 to another number for the hidden size

        # This replaces the standard feed-forward network with an LSTM
        self.latent_dim = 64
        self.share_features_extractor = False

        # Define the actor and critic networks (they are defined in ActorCriticPolicy)
        self.mlp_extractor.policy_net = nn.Linear(self.latent_dim, n_actions)
        self.mlp_extractor.value_net = nn.Linear(self.latent_dim, 1)

    def _forward_lstm(self, obs, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            # Assuming the LSTM is not bidirectional and has a single layer
            hidden_state = torch.zeros(1, obs.size(1), self.latent_dim, device=obs.device)
            cell_state = torch.zeros(1, obs.size(1), self.latent_dim, device=obs.device)

        output, (h, c) = self.lstm(obs, (hidden_state, cell_state))
        return h, c

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        # obs is of shape [batch_size, sequence_length, feature_dim]
        lstm_out, _ = self.lstm(obs)
        # lstm_out is of shape [batch_size, sequence_length, hidden_dim=64]
        features = lstm_out[:, -1, :]
        # features is of shape [batch_size, hidden_dim=64]
        return features

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # Use your custom LSTM feature extraction method
        features = self.extract_features(obs)
        latent_vf = self.forward_critic(features)
        return latent_vf #self.value_net(latent_vf)

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        return self.mlp_extractor.forward_actor(obs)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        return self.mlp_extractor.forward_critic(obs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Using LSTM, we get the sequence of outputs, but we'll only use the last one.
        latent, _ = self._forward_lstm(obs)

        latent_pi = latent[-1].unsqueeze(0)  # Take the last output and add a batch dimension
        latent_vf = latent[-1].unsqueeze(0)  # Take the last output and add a batch dimension

        # Pass through the actor network
        action_logits = self.mlp_extractor.policy_net(latent[:, -1, :])

        # Get the distribution and sample an action
        distribution = Categorical(logits=action_logits)
        if deterministic:
            actions = th.argmax(action_logits, dim=1)
        else:
            actions = distribution.sample()

        # Pass through the critic network
        values = self.mlp_extractor.value_net(latent[:, -1, :])

        # Get the log probability of the actions
        log_prob = distribution.log_prob(actions)

        # Reshape actions to have the same structure as the action space
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=[64, 64])

    def _get_constructor_parameters(self):  # to make the custom policy picklable
        data = super()._get_constructor_parameters()
        data["net_arch"] = self.net_arch
        return data
