from typing import Callable

import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
import torch
from torch import nn

from .encoder_models import create_encoder
from .base_structure import FTA


# Adapted from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class SB3GeneralEncoder(BaseFeaturesExtractor):
  # Uses Imapala architecture for large enough images, smaller model for
  # small images, and small dense layers for single dim observations
  def __init__(self, obs_space: gym.spaces.Box, features_dim: int = 256,
               fta: bool = False, fta_tiles: int = 20):
    if fta:
      super().__init__(obs_space, fta_tiles * features_dim)
    else:
      super().__init__(obs_space, features_dim)

    obs_dim = obs_space.shape
    base_encoder = create_encoder(obs_dim, last_fta=False)
    base_encoder = nn.Sequential(base_encoder, nn.Flatten())

    test_input = torch.zeros(1, *obs_dim)
    with torch.no_grad():
      self.encoder_output_size = base_encoder(test_input).view(-1).shape[0]

    encoder_layers = [
      base_encoder,
      nn.Linear(self.encoder_output_size, features_dim)]

    if fta:
      encoder_layers.append(FTA(features_dim, tiles=fta_tiles))
    else:
      encoder_layers.append(nn.ReLU())
    
    self.encoder = nn.Sequential(*encoder_layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.encoder(obs)

class SB3ActorCriticNetwork(nn.Module):
  def __init__(
      self,
      feature_dim: int,
      hidden_sizes: int = [256],
      policy_fta: bool = False,
      critic_fta: bool = False,
      fta_tiles: int = 20,):
    super().__init__()
    # Save output dimensions, used to create the distributions
    self.latent_dim_pi = hidden_sizes[-1]
    self.latent_dim_vf = hidden_sizes[-1]
    if policy_fta:
      self.latent_dim_pi *= fta_tiles
    if critic_fta:
      self.latent_dim_vf *= fta_tiles
    
    layer_sizes = [feature_dim] + list(hidden_sizes)
    value_layers = []
    policy_layers = []
    for i in range(1, len(layer_sizes)):
      value_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
      policy_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
      
      if not policy_fta or i < len(layer_sizes) - 1:
        policy_layers.append(nn.ReLU())
      if not critic_fta or i < len(layer_sizes) - 1:
        value_layers.append(nn.ReLU())

    if policy_fta:
      policy_layers.append(FTA(layer_sizes[-1], tiles=fta_tiles))
    if critic_fta:
      value_layers.append(FTA(layer_sizes[-1], tiles=fta_tiles))

    self.value_net = nn.Sequential(*value_layers)
    self.policy_net = nn.Sequential(*policy_layers)

  def forward(self, features: torch.Tensor):
    return self.policy_net(features), self.value_net(features)

  def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
    return self.policy_net(features)

  def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
    return self.value_net(features)

class SB3ActorCriticPolicy(ActorCriticPolicy):
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    lr_schedule: Callable[[float], float],
    hidden_sizes=[256],
    policy_fta=False,
    critic_fta=False,
    fta_tiles=20,
    *args,
    **kwargs):

    self.hidden_sizes = hidden_sizes
    self.policy_fta = policy_fta
    self.critic_fta = critic_fta
    self.fta_tiles = fta_tiles

    super().__init__(
      observation_space,
      action_space,
      lr_schedule,
      *args,
      **kwargs)

    # Disable orthogonal initialization
    self.ortho_init = False

  def _build_mlp_extractor(self) -> None:
    self.mlp_extractor = SB3ActorCriticNetwork(
      self.features_dim, self.hidden_sizes, self.policy_fta,
      self.critic_fta, self.fta_tiles)

class SB3QNetwork(QNetwork):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        hidden_sizes=[256],
        fta=False,
        fta_tiles=20,
        **kwargs):
      super().__init__(
        observation_space, action_space, features_extractor,
        features_dim, None, nn.ReLU, False)

      layer_sizes = [features_dim] + list(hidden_sizes) + [action_space.n]
      layers = []
      for i in range(1, len(layer_sizes)):
        layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        if not fta or i < len(layer_sizes) - 1:
          layers.append(nn.ReLU())
      if fta:
        layers.append(FTA(layer_sizes[-1], tiles=fta_tiles))

      self.q_net = nn.Sequential(*layers)

class SB3DQNPolicy(DQNPolicy):
  def __init__(
      self,
      observation_space,
      action_space,
      lr_schedule,
      hidden_sizes=[256],
      fta=False,
      fta_tiles=20,
      **kwargs):
    self.hidden_sizes = hidden_sizes
    self.fta = fta
    self.fta_tiles = fta_tiles

    super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    self.q_net, self.q_net_target = None, None
    self._build(lr_schedule)

  def make_q_net(self) -> QNetwork:
    # Make sure we always have separate networks for features extractors etc
    net_args = self._update_features_extractor(self.net_args, features_extractor=None)
    return SB3QNetwork(**dict(
      hidden_sizes = self.hidden_sizes,
      fta = self.fta,
      fta_tiles = self.fta_tiles,
      **net_args)).to(self.device)