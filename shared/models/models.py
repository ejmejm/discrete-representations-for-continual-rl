import torch
from torch import Tensor, nn
from torch.nn import functional as F

from discrete_mbrl.model_construction import DISCRETE_ENCODER_TYPES
from .base_structure import FTA
from .layers import *


def create_dense_layers(in_features, out_features=128, hidden_sizes=[256, 256]):
  layers = []
  layer_sizes = [in_features] + hidden_sizes + [out_features]
  for i in range(len(layer_sizes) - 1):
    layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    if i < len(layer_sizes) - 2:
      layers.append(nn.ReLU())
  return layers
  
def create_gridworld_layers(n_channels):
  return [
    nn.Conv2d(n_channels, 8, 6, 2),
    nn.ReLU(),
    nn.Conv2d(8, 16, 4, 1),]

def create_gridworld_decoder_layers(n_channels):
  return [
    nn.ReLU(),
    nn.ConvTranspose2d(16, 8, 4, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, n_channels, 6, 2),
    nn.ReLU(),
    nn.Conv2d(n_channels, n_channels, 1, 1)]
  
def create_impala_conv_layers(n_channels):
  return [
    nn.Conv2d(n_channels, 16, 8, 4),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, 2)]

def create_impala_decoder_layers(n_channels):
  return [
    nn.ReLU(),
    nn.ConvTranspose2d(32, 16, 4, 2),
    nn.ReLU(),
    nn.ConvTranspose2d(16, n_channels, 8, 4),
    nn.ReLU(),
    nn.ConvTranspose2d(n_channels, n_channels, 1, 1)]

ACTIVATION_MAP = {
  'relu': nn.ReLU,
  'crelu': CReLU,
  'tanh': nn.Tanh,
}

class ArgmaxLayer(nn.Module):
  def __init__(self, dim=-1):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return x.argmax(dim=self.dim)


def mlp(hidden_sizes, activation='relu', discrete_input=False, n_embeds=None, embed_dim=None):
  if isinstance(activation, str):
    activation_cls = ACTIVATION_MAP[activation.lower()]

  input_multiplier = 1
  if activation == 'crelu':
    input_multiplier = 2

  layers = []

  if discrete_input:
    assert n_embeds is not None, 'Must provide n_embeds if using long_input'
    assert embed_dim is not None, 'Must provide embed_dim if using discrete_input'

    layers.append(ArgmaxLayer(dim=1))
    layers.append(nn.Embedding(n_embeds, embed_dim))
    layers.append(nn.Flatten())
  else:
    layers.append(nn.Flatten())

  for i, (s1, s2) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
    if i > 0:
      s1 = s1 * input_multiplier
    layers.append(nn.Linear(s1, s2))
    layers.append(activation_cls())
  return nn.Sequential(*layers[:-1])

def max_one_hot(logits: Tensor, straight_through_grads: bool = True, dim: int = -1):
  """
  Gets one hot of argmax over logits with straight-through gradients.

  Args:
    logits - shape (..., n_classes, ...)
  """
  indices = logits.argmax(dim=dim)
  ohs = F.one_hot(indices, num_classes=logits.shape[dim])
  if dim != -1 and dim != len(logits.shape) - 1:
    ohs = ohs.transpose(dim, -1)

  if straight_through_grads:
    ohs = ohs + logits - logits.detach()

  return ohs

def sample_one_hot(logits: Tensor, straight_through_grads: bool = True, dim: int = -1):
  """
  Gets one hot samples over catgorical distributions with straight-through gradients.

  Args:
    logits - shape (..., n_classes)
  """
  if dim != -1 and dim != len(logits.shape) - 1:
    logits = logits.transpose(dim, -1)
  probs = F.softmax(logits, dim=-1)
  flat_probs = probs.view(-1, probs.shape[-1])
  samples = torch.multinomial(flat_probs, 1)
  oh_samples = F.one_hot(samples, probs.shape[-1])
  oh_samples = oh_samples.reshape(*probs.shape)

  if straight_through_grads:
    oh_samples = oh_samples + logits - logits.detach()

  if dim != -1 and dim != len(logits.shape) - 1:
    oh_samples = oh_samples.transpose(dim, -1)
  return oh_samples

def logits_to_one_hot(logits: Tensor, stochastic: bool = False,
                      straight_through_grads: bool = True, dim: int = -1):
  if stochastic:
    return sample_one_hot(logits, straight_through_grads, dim=dim)
  return max_one_hot(logits, straight_through_grads, dim=dim)

class OneHotEmbeddings(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embedding_layer = nn.Conv1d(1, self.embedding_dim, self.num_embeddings)

  def forward(self, x):
    """ x is shape (..., num_embeddings) """
    assert x.shape[-1] == self.num_embeddings, \
      f'x.shape[-1] ({x.shape[-1]}) != self.num_embeddings ({self.num_embeddings})'
    rx = x.reshape(-1, 1, x.shape[-1])
    embeddings = self.embedding_layer(rx)
    embeddings = embeddings.reshape(*x.shape[:-1], self.embedding_dim)
    return embeddings

class DuelingDQNModel(nn.Module):
  def __init__(self, obs_dim, n_acts, ae=None, hidden_sizes=None,
               last_fta=False, fta_tiles=20):
    super().__init__()
    if ae is None:
      raise ValueError('Must provide an autoencoder')
      # encoder = create_encoder_from_obs_dim(obs_dim)
    self.n_acts = n_acts
    self.ae = ae
    if ae.encoder_type in DISCRETE_ENCODER_TYPES:
      self.encode = lambda *args, **kwargs: self.ae.encode(*args, **kwargs, as_long=False)
    else:
      self.encode = ae.encode

    test_input = torch.zeros(1, *obs_dim)
    with torch.no_grad():
      self.encoder_output_size = self.encode(test_input).reshape(-1).shape[0]

    if isinstance(hidden_sizes, (int, float)):
      hidden_sizes = [hidden_sizes]
    elif hidden_sizes is None:
      hidden_sizes = []
    layer_sizes = [self.encoder_output_size] + list(hidden_sizes)

    value_layers = [nn.Flatten()]
    adv_layers = [nn.Flatten()]
    for i in range(1, len(layer_sizes)):
      value_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
      adv_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
      
      if not last_fta or i < len(layer_sizes) - 1:
        value_layers.append(nn.ReLU())
        adv_layers.append(nn.ReLU())

    if last_fta:
      value_layers.append(FTA(layer_sizes[-1], tiles=fta_tiles))
      adv_layers.append(FTA(layer_sizes[-1], tiles=fta_tiles))
      layer_sizes.append(layer_sizes[-1] * fta_tiles)

    value_layers.append(nn.Linear(layer_sizes[-1], 1))
    adv_layers.append(nn.Linear(layer_sizes[-1], n_acts))

    self.value_layers = nn.Sequential(*value_layers)
    self.adv_layers = nn.Sequential(*adv_layers)

    self._init_weights()

  def _init_weights(self):
    self.value_layers[-1].weight.data.fill_(0)
    self.value_layers[-1].bias.data.fill_(0)
    self.adv_layers[-1].weight.data.fill_(0)
    self.adv_layers[-1].bias.data.fill_(0)

  def forward_encoded(self, z):
    values = self.value_layers(z)
    advantages = self.adv_layers(z)

    advantage_means = advantages.mean(dim=1, keepdim=True)
    advantages = advantages - advantage_means

    qs = values + advantages

    return qs

  def forward(self, x):
    z = self.encode(x)
    z = z.reshape(x.shape[0], -1) # Flatten
    return self.forward_encoded(z)

  def update_target(self, target_model, tau=1.0):
    for param, target_param in zip(self.parameters(), target_model.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
