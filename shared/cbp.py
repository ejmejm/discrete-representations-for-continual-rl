from collections import defaultdict
import math
import warnings

import torch
from torch import nn


EPSILON = 1e-8


def n_kaiming_uniform(tensor, shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    But has a customizable number of outputs.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return torch.rand(shape) * 2 * bound - bound


class CBPTracker(object):
  def __init__(self, optimizer=None, replace_rate=1e-4, decay_rate=0.99, maturity_threshold=100):
    # Dictionary mapping feature output layer to previous and next layers
    self._tracked_layers = {}
    self._feature_stats = {}
    self._replace_accumulator = defaultdict(float) 

    self.optimizer = optimizer
    
    self.replace_rate = replace_rate
    self.decay_rate = decay_rate
    self.maturity_threshold = maturity_threshold

  def track(self, previous, current, next):
    """Track a list of layers used for CBP calculations."""
    if not isinstance(previous, nn.Linear) or not isinstance(next, nn.Linear):
      raise NotImplementedError('CBP is only implemented for linear layers.')

    self._tracked_layers[current] = (previous, next)
    self._feature_stats[current] = defaultdict(lambda: torch.zeros(1, requires_grad=False))
    current.register_forward_hook(self._get_hook(current))

  def track_sequential(self, sequential):
    """
    Track a sequential model for CBP calculations.
    Must be an alternating sequence of linear layers and activations."""
    for i in range(0, len(sequential) - 2, 2):
      self.track(sequential[i], sequential[i+1], sequential[i+2])

  def track_optimizer(self, optimizer):
    """Track an optimizer for CBP calculations."""
    if self.optimizer is not None:
      warnings.warn('Replacing previously tracked optimizer.')
    self.optimizer = optimizer

  def _get_input_weight_sums(self, layer):
    """Return the sum of the absolute values of the weights for each outputted feature."""
    return torch.sum(torch.abs(layer.weight), dim=1)

  def _get_output_weight_sums(self, layer):
    """Return the sum of the absolute values of the weights for each inputted feature."""
    return torch.sum(torch.abs(layer.weight), dim=0)

  def _reinit_input_weights(self, layer, idxs):
    """Reinitialize the weights that output features at the given indices."""
    # This is how linear layers are initialized in PyTorch
    weight_data = layer.weight.data
    layer.weight.data[idxs] = n_kaiming_uniform(
      weight_data, weight_data[idxs].shape, a=math.sqrt(5))

    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        layer.bias.data[idxs] = torch.rand(len(idxs)) * 2 * bound - bound

  def _reset_input_optim_state(self, layer, idxs):
    """
    Reset the optimizer state for the weights that output features at the given indices.
    Currently works for SGD and Adam (without step reset) optimizers.
    """
    optim_state = self.optimizer.state

    if 'exp_avg' in optim_state[layer.weight]:
      optim_state[layer.weight]['exp_avg'][idxs] = 0
    if 'exp_avg_sq' in optim_state[layer.weight]:
      optim_state[layer.weight]['exp_avg_sq'][idxs] = 0
    if 'momentum_buffer' in optim_state[layer.weight] \
        and optim_state[layer.weight]['momentum_buffer'] is not None:
      optim_state[layer.weight]['momentum_buffer'][idxs] = 0
      
    if layer.bias is not None:
      if 'exp_avg' in optim_state[layer.bias]:
        optim_state[layer.bias]['exp_avg'][idxs] = 0
      if 'exp_avg_sq' in optim_state[layer.bias]:
        optim_state[layer.bias]['exp_avg_sq'][idxs] = 0
      if 'momentum_buffer' in optim_state[layer.bias] \
          and optim_state[layer.bias]['momentum_buffer'] is not None:
        optim_state[layer.bias]['momentum_buffer'][idxs] = 0

  def _reinit_output_weights(self, layer, idxs):
    """Reinitialize the weights that take in features at the given indices."""
    # This is how linear layers are initialized in PyTorch
    weight_data = layer.weight.data
    layer.weight.data[:, idxs] = n_kaiming_uniform(
      weight_data, weight_data[:, idxs].shape, a=math.sqrt(5))

  def _reset_output_optim_state(self, layer, idxs):
    """
    Reset the optimizer state for the weights that take in features at the given indices.
    Currently works for SGD and Adam optimizers.
    """
    optim_state = self.optimizer.state

    if 'exp_avg' in optim_state[layer.weight]:
      optim_state[layer.weight]['exp_avg'][:, idxs] = 0
    if 'exp_avg_sq' in optim_state[layer.weight]:
      optim_state[layer.weight]['exp_avg_sq'][:, idxs] = 0
    if 'momentum_buffer' in optim_state[layer.weight] \
        and optim_state[layer.weight]['momentum_buffer'] is not None:
      optim_state[layer.weight]['momentum_buffer'][:, idxs] = 0

  def _get_hook(self, layer):
    """Return a hook function for a given layer."""
    def track_cbp_stats(module, input, output):
      if not module.training:
        return

      input = input[0]
      output = output[-1]
      
      with torch.no_grad():
        # a
        self._feature_stats[module]['age'] = torch.ones_like(output) + self._feature_stats[module]['age']

        # f
        self._feature_stats[module]['running_avg'] = (1 - self.decay_rate) * output \
          + self.decay_rate * self._feature_stats[module]['running_avg']

        # f_hat
        scaled_avg = self._feature_stats[layer]['running_avg'] \
          / (1 - self.decay_rate ** self._feature_stats[layer]['age'])

        # y
        self._feature_stats[module]['utility'] = \
          (torch.abs(output - scaled_avg) \
          * self._get_output_weight_sums(self._tracked_layers[layer][1])) \
          / (self._get_input_weight_sums(self._tracked_layers[layer][0]) + EPSILON)

        # u
        self._feature_stats[module]['avg_utility'] = \
          (1 - self.decay_rate) * self._feature_stats[module]['utility'] \
          + self.decay_rate * self._feature_stats[module]['avg_utility']

        # u_hat
        self._feature_stats[module]['scaled_avg_utility'] = \
          self._feature_stats[module]['avg_utility'] \
          / (1 - self.decay_rate ** self._feature_stats[module]['age'])

    return track_cbp_stats

  def _reset_feature_stats(self, layer, idxs):
    """Resets the feature stats for the given layer and indices."""
    for key in self._feature_stats[layer]:
      self._feature_stats[layer][key][idxs] = 0

  def _prune_layer(self, layer):
    ages = self._feature_stats[layer]['age']

    # Get number of features to reset
    n_features = ages.numel()
    self._replace_accumulator[layer] += self.replace_rate * n_features

    # If there are not enough features to reset, return
    if self._replace_accumulator[layer] < 1:
      return

    # Get eligible features
    eligible_idxs = ages > self.maturity_threshold
    eligible_idxs = torch.nonzero(eligible_idxs).squeeze()

    # If there are no eligible features, return
    if eligible_idxs.numel() == 0:
      return

    # print(f'Replace {self._replace_accumulator[layer]} / {n_features}')

    # Get features to reset based on lowest utility
    n_reset = int(self._replace_accumulator[layer])
    self._replace_accumulator[layer] -= n_reset
    reset_idxs = torch.argsort(
      self._feature_stats[layer]['scaled_avg_utility'][eligible_idxs])[:n_reset]
    reset_idxs = eligible_idxs[reset_idxs]
    
    # Reset feature stats
    self._reset_feature_stats(layer, reset_idxs)

    # Reset features
    self._reinit_input_weights(self._tracked_layers[layer][0], reset_idxs)
    self._reinit_output_weights(self._tracked_layers[layer][1], reset_idxs)

    # Reset optimizer state
    if self.optimizer is not None:
      self._reset_input_optim_state(self._tracked_layers[layer][0], reset_idxs)
      self._reset_output_optim_state(self._tracked_layers[layer][1], reset_idxs)

    return reset_idxs

  def prune_features(self):
    """Prune features based on the CBP score."""
    reset_idxs = {}
    for layer in self._tracked_layers.keys():
      layer_idxs = self._prune_layer(layer)
      if layer_idxs is not None:
        reset_idxs[layer] = layer_idxs
    return reset_idxs