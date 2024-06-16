from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Source: https://github.com/hwang-ua/fta_pytorch_implementation/blob/main/core/lta.py
class FTA(nn.Module):
    def __init__(self, input_dim, tiles=20, bound_low=-2, bound_high=2, eta=0.2):
        super(FTA, self).__init__()
        # 1 tiling, binning
        self.n_tilings = 1
        self.n_tiles = tiles
        self.bound_low, self.bound_high = bound_low, bound_high
        self.delta = (self.bound_high - self.bound_low) / self.n_tiles
        c_mat = torch.as_tensor(np.array([self.delta * i for i in range(self.n_tiles)]) \
            + self.bound_low, dtype=torch.float32)
        self.register_buffer('c_mat', c_mat)
        self.eta = eta
        self.d = input_dim

    def forward(self, reps):
        temp = reps
        temp = temp.reshape([-1, self.d, 1])
        onehots = 1.0 - self.i_plus_eta(self.sum_relu(self.c_mat, temp))
        out = torch.reshape(torch.reshape(onehots, [-1]), [-1, int(self.d * self.n_tiles * self.n_tilings)])
        return out

    def sum_relu(self, c, x):
        out = F.relu(c - x) + F.relu(x - self.delta - c)
        return out

    def i_plus_eta(self, x):
        if self.eta == 0:
            return torch.sign(x)
        out = (x <= self.eta).type(torch.float32) * x + (x > self.eta).type(torch.float32)
        return out


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, *self.shape)


class ExternalModule(nn.Module):
    """
    This is a wrapper for external modules that are not part of the model.
    It is used to make sure that the model is not saved with the external module,
    nor will they be passed to the optimizer.
    """
    def __init__(self, layer):
        super(ExternalModule, self).__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)

    def named_parameters(self, prefix='', recurse=True):
        return iter(())

    def parameters(self, recurse=True):
        return iter(())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return OrderedDict()


def create_simple_1D_encoder(input_dim):
    flat_dim = input_dim[0] * input_dim[1]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU())

def create_simple_1D_decoder(input_dim):
    flat_dim = input_dim[0] * input_dim[1]
    return nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, flat_dim),
        ReshapeLayer(input_dim))

def create_gridworld_encoder(n_channels=1):
    return nn.Sequential(
        nn.Conv2d(n_channels, 8, 4, 2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU())

def create_gridworld_decoder(n_channels=1):
    return nn.Sequential(
        nn.ConvTranspose2d(16, 8, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, n_channels, 4, 2),
        nn.Conv2d(n_channels, n_channels, 3, 1, 1))

def create_atari_encoder(n_channels=4):
    return nn.Sequential(
        nn.Conv2d(n_channels, 32, 5, 5, 0),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, 5, 0),
        nn.ReLU())

def create_atari_decoder(n_channels=4):
    return nn.Sequential(
        nn.ConvTranspose2d(64, 32, 5, 5, 0),
        nn.ReLU(),
        nn.ConvTranspose2d(32, n_channels, 5, 5, 0),
        nn.ReLU(),
        nn.Conv2d(n_channels, n_channels, 3, 1, 1))

GYM_HIDDEN_SIZE = 32
GRIDWORLD_HIDDEN_SIZE = 64
ATARI_HIDDEN_SIZE = 256

def get_hidden_size_from_obs_dim(obs_dim):
    if len(obs_dim) == 2:
        if obs_dim[0] <= 32:
            return GYM_HIDDEN_SIZE
        else:
            raise Exception('1D observation dimensions this large are not supported!')
    elif obs_dim[1] <= 32:
        return GRIDWORLD_HIDDEN_SIZE
    return ATARI_HIDDEN_SIZE

def create_encoder_from_obs_dim(obs_dim):
    if len(obs_dim) == 2:
        if obs_dim[0] <= 32:
            return create_simple_1D_encoder(obs_dim)
        else:
            raise Exception('1D observation dimensions this large are not supported!')
    elif obs_dim[1] <= 32:
        return create_gridworld_encoder(obs_dim[0])
    return create_atari_encoder(obs_dim[0])

def create_decoder_from_obs_dim(obs_dim):
    if len(obs_dim) == 1:
        if obs_dim[0] <= 32:
            return create_simple_1D_decoder(obs_dim)
        else:
            raise Exception('1D observation dimensions this large are not supported!')
    elif obs_dim[1] <= 32:
        return create_gridworld_decoder(obs_dim[0])
    return create_atari_decoder(obs_dim[0])