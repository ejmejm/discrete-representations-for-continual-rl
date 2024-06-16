from abc import ABC, abstractmethod
from collections import OrderedDict

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


EPSILON = 1e-8


def one_hot_cross_entropy(pred, target):
  """ Calculate the cross entropy between two one-hot vectors. """
  return -torch.sum(target * torch.log(pred + EPSILON), dim=-1)


def get_main_trans_layers(model):
  layers = []
    
  for module in model.shared_layers:
    if isinstance(module, nn.Linear):
      layers.append(module)
          
  for module in model.state_head:
    if isinstance(module, nn.Linear):
      layers.append(module)
          
  return layers

def get_main_trans_activations(model):
  layers = []
    
  for module in model.shared_layers:
    if isinstance(module, nn.ReLU):
      layers.append(module)
          
  for module in model.state_head:
    if isinstance(module, nn.ReLU):
      layers.append(module)
          
  return layers


def record_trans_model_update(trans_model, loss, optimizer, activations=None, grad_clip=0):
  modules = get_main_trans_layers(trans_model)
  norms = {}

  # Compute gradients
  loss.backward()
  if grad_clip > 0:
    nn.utils.clip_grad_norm_(trans_model.parameters(), grad_clip)

  # Store the weights before the update
  weights_before_update = [module.weight.data.clone() for module in modules]

  # Update the weights
  optimizer.step()

  grads_flattened = torch.cat([module.weight.grad.view(-1) for module in modules])

  # l0, l1, and l2 norms of the gradients
  grad_l0_norm = torch.norm(grads_flattened, p=0)
  grad_l1_norm = torch.norm(grads_flattened, p=1)
  grad_l2_norm = torch.norm(grads_flattened, p=2)

  # l2 norm of the weight update
  weight_update_flattened = torch.cat(
    [(module.weight.data - weight_before).view(-1)
      for module, weight_before in zip(modules, weights_before_update)])
  weight_change_l2_norm = torch.norm(weight_update_flattened, p=2)

  norms['grad_l0_norm'] = grad_l0_norm.item()
  norms['grad_l1_norm'] = grad_l1_norm.item()
  norms['grad_l2_norm'] = grad_l2_norm.item()
  norms['weight_change_l2_norm'] = weight_change_l2_norm.item()

  if activations is not None:
    activations = torch.concat([
      a.view(-1) for a in activations])
    norms['activation_l0_norm'] = torch.norm(activations, p=0).item()

  return norms


class ActivationRecorder:
  def __init__(self, modules: nn.Module):
    self.activations = []
    self._hooks = []
    for module in modules:
      hook = module.register_forward_hook(self.record_activation)
      self._hooks.append(hook)

  def record_activation(self, module, input, output):
    self.activations.append(output.detach())

  def reset(self):
    out = self.activations
    self.activations = []
    return out


class BaseRepresentationLearner(ABC):
  def __init__(self, model=None, batch_size=32, update_freq=32, log_freq=100):
    if model is None:
      self._init_model()
    else:
      self.model = model

    assert hasattr(self.model, 'encoder'), \
      'Model must have an encoder!'

    self.encoder = self.model.encoder
    self.batch_size = batch_size
    self.update_freq = update_freq
    self.log_freq = log_freq

  @abstractmethod
  def _init_model(self, *args, **kwargs):
    pass

  @abstractmethod
  def calculate_losses(self, batch_data):
    pass

  @abstractmethod
  def train(self, batch_data):
    pass

class AETrainer(BaseRepresentationLearner):
  def __init__(
      self,
      model: nn.Module,
      batch_size: int = 256,
      update_freq: int = 128,
      log_freq: int = 100,
      lr: float = 3e-4,
      recon_loss_clip: float = 0,
      grad_clip: float = 0):
    super().__init__(model, batch_size, update_freq, log_freq)
    self.model = model
    self.recon_loss_clip = recon_loss_clip
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.train_step = 0
    self.grad_clip = grad_clip
    
  def _init_model(self):
    raise Exception('VAE requires a model to be specified!')

  def calculate_losses(self, batch_data, return_stats=False):
    loss_dict = {}
    device = next(self.model.parameters()).device
    sample_size = int(batch_data[0].shape[0] / 2)
    obs = torch.cat(
      [batch_data[0][:sample_size],
       batch_data[2][sample_size:]], dim=0).to(device)
    
    if self.model.encoder_type == 'fta_ae':
      obs_recon, latent_means, _ = self.model(obs, return_all=True)
    else:
      obs_recon = self.model(obs)

    recon_loss = (obs - obs_recon) ** 2
    if self.recon_loss_clip > 0:
      recon_loss = torch.max(recon_loss, torch.tensor(self.recon_loss_clip, device=device))
    recon_loss = recon_loss.reshape(recon_loss.shape[0], -1).sum(-1)
    loss_dict['recon_loss'] = recon_loss.mean()

    if return_stats:
      stats = {}

      # Count number of latent in range (where latent is not all 0)
      if self.model.encoder_type == 'fta_ae':
        flat_latents = latent_means.reshape(latent_means.shape[0], -1)
        non_zero = torch.clip(torch.sum(flat_latents != 0, dim=1), 0, 1)
        mean_non_zero = torch.mean(non_zero.float())
        stats['non_zero_latent_frac'] = mean_non_zero

      return loss_dict, stats
    
    return loss_dict

  def train(self, batch_data):
    loss_dict, stats = self.calculate_losses(batch_data, return_stats=True)
    loss = torch.sum(torch.stack(tuple(loss_dict.values())))

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      log_str = f'AE train step {self.train_step} | Loss: {loss.item():.4f}'
      for loss_name, loss_value in loss_dict.items():
        log_str += f' | {loss_name}: {loss_value.item():.4f}'
      print(log_str)

    self.optimizer.zero_grad()
    loss.backward()
    if self.grad_clip > 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    self.optimizer.step()
    self.train_step += 1

    return loss_dict, stats

class VAETrainer(BaseRepresentationLearner):
  def __init__(
      self,
      model: nn.Module,
      batch_size: int = 256,
      update_freq: int = 128,
      log_freq: int = 100,
      lr: float = 3e-4,
      recon_loss_clip: float = 0,
      grad_clip: float = 0):
    super().__init__(model, batch_size, update_freq, log_freq)
    self.model = model
    self.recon_loss_clip = recon_loss_clip
    self.grad_clip = grad_clip
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.train_step = 0
    
  def _init_model(self):
    raise Exception('VAE requires a model to be specified!')

  def calculate_losses(self, batch_data):
    device = next(self.model.parameters()).device
    sample_size = int(batch_data[0].shape[0] / 2)
    obs = torch.cat(
      [batch_data[0][:sample_size],
       batch_data[2][sample_size:]], dim=0).to(device)
    obs_recon, mu, sigma = self.model(obs, return_all=True)

    kl_div = 0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2)
    kl_div = kl_div.view(kl_div.shape[0], -1).sum(-1)

    recon_loss = (obs - obs_recon) ** 2
    if self.recon_loss_clip > 0:
      recon_loss = torch.max(recon_loss, torch.tensor(self.recon_loss_clip, device=device))
    recon_loss = recon_loss.reshape(recon_loss.shape[0], -1).sum(-1)

    losses = -kl_div + recon_loss
    return losses

  def train(self, batch_data):
    losses = self.calculate_losses(batch_data)
    loss = losses.mean()

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      print(f'VAE train step {self.train_step} | Loss: {loss.item():.4f}')

    self.optimizer.zero_grad()
    loss.backward()
    if self.grad_clip > 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    self.optimizer.step()
    self.train_step += 1

    vis_loss = loss.item() - self.recon_loss_clip * np.prod(batch_data[0].shape[1:])
    return vis_loss, {}

class VQVAETrainer(BaseRepresentationLearner):
  def __init__(
      self,
      model: nn.Module,
      batch_size: int = 256,
      update_freq: int = 128,
      log_freq: int = 100,
      lr: float = 3e-4,
      recon_loss_clip: float = 0,
      grad_clip: float = 0):
    super().__init__(model, batch_size, update_freq, log_freq)
    self.model = model
    self.recon_loss_clip = recon_loss_clip
    self.grad_clip = grad_clip
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.train_step = 0
    self.mi_coefs = torch.linspace(0, 0.002, 2000)
    
  def _init_model(self):
    raise Exception('VQVAE requires a model to be specified!')

  def calculate_losses(self, batch_data):
    device = next(self.model.parameters()).device
    sample_size = int(batch_data[0].shape[0] / 2)
    obs = torch.cat(
      [batch_data[0][:sample_size],
       batch_data[2][sample_size:]], dim=0).to(device)
    loss_dict = {}
    obs_recon, quantizer_loss, perplexity, oh_encodings = self.model(obs)
    loss_dict['quantizer_loss'] = quantizer_loss

    # Recon loss
    recon_loss = (obs - obs_recon) ** 2
    if self.recon_loss_clip > 0:
      recon_loss = torch.max(recon_loss, torch.tensor(self.recon_loss_clip, device=device))
    recon_loss = recon_loss.reshape(recon_loss.shape[0], -1).sum(-1)
    recon_loss = recon_loss.mean()
    loss_dict['recon_loss'] = recon_loss

    return loss_dict

  def train(self, batch_data): # Batch data is (s, a, s', r, d) tensors
    loss_dict = self.calculate_losses(batch_data)
    loss = torch.sum(torch.stack(tuple(loss_dict.values())))

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      log_str = f'VQVAE train step {self.train_step} | Loss: {loss.item():.4f}'
      for loss_name, loss_value in loss_dict.items():
        log_str += f' | {loss_name}: {loss_value.item():.4f}'
      print(log_str)

    self.optimizer.zero_grad()
    loss.backward()
    if self.grad_clip > 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    self.optimizer.step()
    self.train_step += 1

    # vis_loss = loss.item() - self.recon_loss_clip * np.prod(batch_data[0].shape[1:])
    return loss_dict, {}

class DiscreteTransitionTrainer():
  def __init__(
      self,
      transition_model: nn.Module,
      encoder: nn.Module,
      log_freq: int = 100,
      log_norms: bool = False,
      lr: float = 1e-3,
      incl_encoder=False,
      grad_clip: float = 0,):
    self.model = transition_model
    self.encoder = encoder
    self.log_freq = log_freq
    self.log_norms = log_norms
    self.train_step = 0
    self.default_gamma = 0.99
    self.grad_clip = grad_clip
    self.incl_encoder = incl_encoder

    if self.incl_encoder:
      self.optimizer = optim.Adam(
        list(self.model.parameters()) + list(self.encoder.parameters()), lr=lr)
    else:
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    if self.log_norms:
      self.activation_recorder = ActivationRecorder(
        get_main_trans_activations(self.model))

  def _init_model(self):
    raise Exception('DiscreteTransitionTrainer requires a model to be specified!')

  def calculate_accuracy(self, batch_data):
    device = next(self.model.parameters()).device
    obs = batch_data[0].to(device)
    acts = batch_data[1].to(device)
    next_obs = batch_data[2].to(device)

    with torch.no_grad():
      encodings = self.encoder.encode(obs)
      next_encodings = self.encoder.encode(next_obs)
    pred_next_encodings = self.model(encodings, acts)
    accuracy = (next_encodings == pred_next_encodings).float().mean()
    return accuracy

  def calculate_losses(self, batch_data, n=1):
    if n == 1:
      batch_data = [x.unsqueeze(1) for x in batch_data]
    assert batch_data[0].shape[1] == n, 'n steps does not match batch size!'

    device = next(self.model.parameters()).device

    initial_obs = batch_data[0][:, 0].to(device)

    encodings = self.encoder.encode(initial_obs)
    if not self.incl_encoder:
      encodings = encodings.detach()

    losses = OrderedDict()
    batch_size = initial_obs.shape[0]
    loss_mask = torch.ones(batch_size, device=device, requires_grad=False)
    for i in range(n):
      acts = batch_data[1][:, i].to(device)
      next_obs = batch_data[2][:, i].to(device)
      rewards = batch_data[3][:, i].to(device)
      dones = batch_data[4][:, i].to(device)
      gammas = (1 - dones.float()) * self.default_gamma

      with torch.no_grad():
        next_encodings = self.encoder.encode(next_obs)
        
      oh_outcomes = None
      if self.model.stochastic == 'categorical':
        oh_outcomes, outcome_logits = self.model.discretize(next_encodings, return_logits=True)
        # TODO: Update this if I ever need to add more variable types to the replay buffer
        if len(batch_data) > 5:
          target_outcomes = batch_data[5][:, i].long().to(device)
          state_disc_loss = F.cross_entropy(outcome_logits, target_outcomes, reduction='none')
          losses[f'{i+1}_step_state_disc_loss'] = state_disc_loss.masked_select(
            loss_mask.bool()).mean()
          oh_outcomes = oh_outcomes.detach()
        
      next_logits_pred, reward_preds, gamma_preds, stoch_logits = self.model(
        encodings, acts, oh_outcomes=oh_outcomes, return_logits=True, return_stoch_logits=True)

      if self.model.stochastic == 'categorical':
        stoch_probs = F.softmax(stoch_logits, dim=1)
        outcome_losses = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
        losses[f'{i+1}_step_outcome_loss'] = outcome_losses.masked_select(loss_mask[:, None].bool()).mean()

      ### State Loss ##
      state_loss = F.cross_entropy(
        next_logits_pred, next_encodings, reduction='none')
      state_loss = state_loss.view(state_loss.shape[0], -1).sum(dim=1)
      state_loss = state_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_state_loss'] = state_loss.mean()

      ### Reward Loss ###
      reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
      reward_loss = reward_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_reward_loss'] = reward_loss.mean()

      ### Gamma Loss ###
      gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
      gamma_loss = gamma_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_gamma_loss'] = gamma_loss.mean()

      with torch.no_grad():
        mask_changes = dones.float().nonzero().squeeze()
        loss_mask.scatter_(0, mask_changes, 0)

      encodings = next_logits_pred.argmax(dim=1).detach()

    return losses

  def train(self, batch_data, n=1):
    loss_dict = self.calculate_losses(batch_data, n)
    loss = torch.sum(torch.stack(tuple(loss_dict.values())))

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      log_str = f'DTransModel train step {self.train_step} | Loss: {loss.item():.4f}'
      for loss_name, loss_value in loss_dict.items():
        log_str += f' | {loss_name}: {loss_value.item():.4f}'
      print(log_str)

    self.train_step += 1

    self.optimizer.zero_grad()

    if self.log_norms:
      norm_data = record_trans_model_update(
        self.model, loss, self.optimizer, self.activation_recorder.reset(), self.grad_clip)
    else:
      norm_data = {}
      loss.backward()
      if self.grad_clip > 0:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()
      
    return loss_dict, norm_data


class UniversalVQTransitionTrainer():
  def __init__(
      self,
      transition_model: nn.Module,
      encoder: nn.Module,
      log_freq: int = 100,
      log_norms: bool = False,
      lr: float = 1e-3,
      incl_encoder=False,
      loss_type='cross_entropy',
      grad_clip: float = 0,):
    self.model = transition_model
    self.encoder = encoder
    self.log_freq = log_freq
    self.log_norms = log_norms
    self.train_step = 0
    self.default_gamma = 0.99
    self.grad_clip = grad_clip
    self.incl_encoder = incl_encoder
    self.use_rand_mask = getattr(
      transition_model, 'rand_mask', None) is not None

    if self.incl_encoder:
      self.optimizer = optim.Adam(
        list(self.model.parameters()) + list(self.encoder.parameters()), lr=lr)
    else:
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    if loss_type == 'cross_entropy':
      assert encoder.encoder_type == 'vqvae', \
        'Cross entropy loss requires a VQVAE encoder!'
      self.loss_fn = F.cross_entropy
    elif loss_type == 'mse':
      self.loss_fn = F.mse_loss
    else:
      raise Exception(f'Unknown loss type: {loss_type}')

    if self.log_norms:
      self.activation_recorder = ActivationRecorder(
        get_main_trans_activations(self.model))

  def _init_model(self):
    raise Exception('DiscreteTransitionTrainer requires a model to be specified!')

  def calculate_accuracy(self, batch_data):
    device = next(self.model.parameters()).device
    obs = batch_data[0].to(device)
    acts = batch_data[1].to(device)
    next_obs = batch_data[2].to(device)

    with torch.no_grad():
      encodings = self.encoder.encode(obs)
      next_encodings = self.encoder.encode(next_obs)
    pred_next_encodings = self.model(encodings, acts)
    comparison = next_encodings == pred_next_encodings
    comparison = comparison.view(comparison.shape[0], -1).all(dim=1)
    accuracy = comparison.float().mean()
    return accuracy

  def calculate_losses(self, batch_data, n=1):
    if n == 1:
      batch_data = [x.unsqueeze(1) for x in batch_data]
    assert batch_data[0].shape[1] == n, 'n steps does not match batch size!'

    device = next(self.model.parameters()).device

    initial_obs = batch_data[0][:, 0].to(device)

    encodings = self.encoder.encode(initial_obs)
    if not self.incl_encoder:
      encodings = encodings.detach()

    losses = OrderedDict()
    batch_size = initial_obs.shape[0]
    loss_mask = torch.ones(batch_size, device=device, requires_grad=False)
    for i in range(n):
      acts = batch_data[1][:, i].to(device)
      next_obs = batch_data[2][:, i].to(device)
      rewards = batch_data[3][:, i].to(device)
      dones = batch_data[4][:, i].to(device)
      gammas = (1 - dones.float()) * self.default_gamma

      with torch.no_grad():
        next_encodings = self.encoder.encode(next_obs)
        
      oh_outcomes = None
      if self.model.stochastic == 'categorical':
        oh_outcomes, outcome_logits = self.model.discretize(next_encodings, return_logits=True)
        # TODO: Update this if I ever need to add more variable types to the replay buffer
        if len(batch_data) > 5:
          target_outcomes = batch_data[5][:, i].long().to(device)
          state_disc_loss = F.cross_entropy(outcome_logits, target_outcomes, reduction='none')
          losses[f'{i+1}_step_state_disc_loss'] = state_disc_loss.masked_select(
            loss_mask.bool()).mean()
          oh_outcomes = oh_outcomes.detach()
        
      next_logits_pred, reward_preds, gamma_preds, stoch_logits = self.model(
        encodings, acts, oh_outcomes=oh_outcomes, return_logits=True, return_stoch_logits=True)

      if self.model.stochastic == 'categorical':
        stoch_probs = F.softmax(stoch_logits, dim=1)
        outcome_losses = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
        losses[f'{i+1}_step_outcome_loss'] = outcome_losses.masked_select(loss_mask[:, None].bool()).mean()

      ### State Loss ##
      if self.loss_fn == F.cross_entropy:
        state_loss = F.cross_entropy(
          next_logits_pred, next_encodings, reduction='none')
        
      elif self.loss_fn == F.mse_loss:
        # one-hot conversion if needed (for hard vqvae)
        if next_encodings.dtype == torch.long:
          next_encodings = F.one_hot(
            next_encodings, num_classes=next_logits_pred.shape[1])
          next_encodings = rearrange(next_encodings, 'b ... c -> b c ...')
          next_encodings = next_encodings.float()

        if self.use_rand_mask:
          next_encodings = next_encodings * self.model.rand_mask[None]

        state_loss = F.mse_loss(
          next_logits_pred, next_encodings, reduction='none')

      state_loss = state_loss.view(state_loss.shape[0], -1).sum(dim=1)
      state_loss = state_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_state_loss'] = state_loss.mean()

      ### Reward Loss ###
      reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
      reward_loss = reward_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_reward_loss'] = reward_loss.mean()

      ### Gamma Loss ###
      gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
      gamma_loss = gamma_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_gamma_loss'] = gamma_loss.mean()

      with torch.no_grad():
        mask_changes = dones.float().nonzero().squeeze()
        loss_mask.scatter_(0, mask_changes, 0)
        encodings = self.model.logits_to_state(next_logits_pred.detach())

    return losses

  def train(self, batch_data, n=1):
    loss_dict = self.calculate_losses(batch_data, n)
    loss = torch.sum(torch.stack(tuple(loss_dict.values())))

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      log_str = f'Universal VQ Trans Model train step {self.train_step} | Loss: {loss.item():.4f}'
      for loss_name, loss_value in loss_dict.items():
        log_str += f' | {loss_name}: {loss_value.item():.4f}'
      print(log_str)

    self.train_step += 1

    self.optimizer.zero_grad()

    if self.log_norms:
      norm_data = record_trans_model_update(
        self.model, loss, self.optimizer, self.activation_recorder.reset(), self.grad_clip)
    else:
      norm_data = {}
      loss.backward()
      if self.grad_clip > 0:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()
      
    return loss_dict, norm_data
  

class ContinuousTransitionTrainer():
  def __init__(
      self,
      transition_model: nn.Module,
      encoder: nn.Module,
      log_freq: int = 100,
      log_norms: bool = False,
      lr: float = 1e-3,
      grad_clip: float = 0,
      e2e_loss: bool = False,
  ):
    self.model = transition_model
    self.encoder = encoder
    self.log_freq = log_freq
    self.log_norms = log_norms
    self.train_step = 0
    self.default_gamma = 0.99
    self.grad_clip = grad_clip
    self.e2e_loss = e2e_loss
    
    if self.e2e_loss:
      self.optimizer = optim.Adam(
        list(self.model.parameters()) + list(self.encoder.parameters()), lr=lr)
    else:
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    if self.log_norms:
      self.activation_recorder = ActivationRecorder(
        get_main_trans_activations(self.model))

  def calculate_losses(self, batch_data, n=1):
    if n == 1:
      batch_data = [x.unsqueeze(1) for x in batch_data]

    device = next(self.model.parameters()).device

    initial_obs = batch_data[0][:, 0].to(device)

    encodings = self.encoder.encode(initial_obs)
    if not self.e2e_loss:
      encodings = encodings.detach()

    losses = OrderedDict()
    batch_size = initial_obs.shape[0]
    loss_mask = torch.ones(batch_size, device=device, requires_grad=False)
    for i in range(n):
      acts = batch_data[1][:, i].to(device)
      next_obs = batch_data[2][:, i].to(device)
      rewards = batch_data[3][:, i].to(device)
      dones = batch_data[4][:, i].to(device)
      gammas = (1 - dones.float()) * self.default_gamma

      with torch.no_grad():
        next_encodings = self.encoder.encode(next_obs, as_long=False)

      # Required for encoders like soft_vqvae that are not flat by default
      next_encodings = next_encodings.reshape(
        next_obs.shape[0], self.encoder.latent_dim)

      oh_outcomes = None
      if self.model.stochastic == 'categorical':
        oh_outcomes, outcome_logits = self.model.discretize(next_encodings, return_logits=True)
        # TODO: Update this if I ever need to add more variable types to the replay buffer
        if len(batch_data) > 5:
          target_outcomes = batch_data[5][:, i].long().to(device)
          state_disc_loss = F.cross_entropy(outcome_logits, target_outcomes, reduction='none')
          losses[f'{i+1}_step_state_disc_loss'] = state_disc_loss.masked_select(
            loss_mask.bool()).mean()
          oh_outcomes = oh_outcomes.detach()

      next_encodings_pred, reward_preds, gamma_preds, stoch_logits = \
        self.model(encodings, acts, oh_outcomes=oh_outcomes, return_logits=True, return_stoch_logits=True)

      if self.model.stochastic == 'categorical':
        stoch_probs = F.softmax(stoch_logits, dim=1)
        outcome_losses = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
        losses[f'{i+1}_step_outcome_loss'] = outcome_losses.masked_select(
          loss_mask[:, None].bool()).mean()

      if self.e2e_loss:
        # Calculate obs recon loss for e2e training
        obs_recon = self.encoder.decode(next_encodings_pred)
        recon_loss = F.mse_loss(next_obs, obs_recon, reduction='none').reshape(
          next_obs.shape[0], -1)
        losses[f'{i+1}_step_recon_loss'] = recon_loss.masked_select(
          loss_mask[:, None].bool()).reshape(-1, recon_loss.shape[1]).sum(dim=1).mean()
        
      # Calculate the MSE losses for transition only training
      state_loss = F.mse_loss(
        next_encodings_pred.view(next_obs.shape[0], self.encoder.latent_dim),
        next_encodings, reduction='none')
      losses[f'{i+1}_step_state_loss'] = state_loss.masked_select(
        loss_mask[:, None].bool()).mean()
      if self.e2e_loss:
        losses[f'{i+1}_step_state_loss'] = losses[f'{i+1}_step_state_loss'].detach()

      reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
      losses[f'{i+1}_step_reward_loss'] = reward_loss.masked_select(
        loss_mask.bool()).mean()

      gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
      losses[f'{i+1}_step_gamma_loss'] = gamma_loss.masked_select(
        loss_mask.bool()).mean()

      mask_changes = dones.float().nonzero().squeeze()
      loss_mask.scatter_(0, mask_changes, 0)
      encodings = self.model.logits_to_state(next_encodings_pred.detach())
      # TODO: Experiment with whether I should detach this either way
      if not self.e2e_loss:
        encodings = encodings.detach()

    return losses

  def train(self, batch_data, n=1):
    loss_dict = self.calculate_losses(batch_data, n)
    loss = torch.sum(torch.stack(tuple(loss_dict.values())))

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      log_str = f'CTransModel train step {self.train_step} | Loss: {loss.item():.4f}'
      for loss_name, loss_value in loss_dict.items():
        log_str += f' | {loss_name}: {loss_value.item():.4f}'
      print(log_str)

    self.train_step += 1

    self.optimizer.zero_grad()

    if self.log_norms:
      norm_data = record_trans_model_update(
        self.model, loss, self.optimizer, self.activation_recorder.reset(), self.grad_clip)
    else:
      norm_data = {}
      loss.backward()
      if self.grad_clip > 0:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()

    return loss_dict, norm_data

# Not being updated
class TransformerTransitionTrainer():
  def __init__(
      self,
      transition_model: nn.Module,
      encoder: nn.Module,
      log_freq: int = 100,
      lr: float = 1e-3,
      grad_clip: float = 0,):
    self.model = transition_model
    self.encoder = encoder
    self.log_freq = log_freq
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.train_step = 0
    self.default_gamma = 0.99
    self.grad_clip = grad_clip

  def calculate_losses(self, batch_data, n=1):
    """
    Args:
        batch_data: List, of shape [5, batch_size, n_steps, ...]
        n: int, number of steps to train on
    """
    if n == 1:
      batch_data = [x.unsqueeze(1) for x in batch_data]

    device = next(self.model.parameters()).device

    initial_obs = batch_data[0][:, 0].to(device)
    with torch.no_grad():
        encodings = self.encoder.encode(initial_obs)

    losses = OrderedDict()
    batch_size = initial_obs.shape[0]
    loss_mask = torch.ones(batch_size, device=device, requires_grad=False)
    for i in range(n):
      acts = batch_data[1][:, i].to(device)
      next_obs = batch_data[2][:, i].to(device)
      rewards = batch_data[3][:, i].to(device)
      dones = batch_data[4][:, i].to(device)
      gammas = (1 - dones.float()) * self.default_gamma

      with torch.no_grad():
        next_encodings = self.encoder.encode(next_obs)
      
      sequence_length = next_encodings.shape[1]
      if self.model.model_type.lower() == 'transformer':
        mask = self.model.get_tgt_mask(sequence_length).to(device)
      elif self.model.model_type.lower() == 'transformerdec':
        mask = self.model.get_tgt_mask(encodings.shape[1] + 1, sequence_length).to(device)
      else:
        raise ValueError(f'Unknown model type: {self.model.model_type}')

      if self.model.training:
        next_logits_pred, reward_preds, gamma_preds = self.model(
          encodings, acts, next_encodings, tgt_mask=mask, return_logits=True)
      else:
        next_logits_pred, reward_preds, gamma_preds = self.model(
          encodings, acts, return_logits=True)

      # Calculate the loss with categorical cross entropy
      state_loss = F.cross_entropy(
        next_logits_pred.permute(0, 2, 1), next_encodings, reduction='none')

      if not self.model.training:
        probs = F.softmax(next_logits_pred, dim=-1)

      state_loss = state_loss.view(state_loss.shape[0], -1).sum(dim=1)
      state_loss = state_loss.masked_select(loss_mask.bool())
      losses[f'{i+1}_step_state_loss'] = state_loss.mean()

      # reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
      # reward_loss = reward_loss.masked_select(loss_mask.bool())
      # losses[f'{i+1}_step_reward_loss'] = reward_loss.mean()

      # gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
      # gamma_loss = gamma_loss.masked_select(loss_mask.bool())
      # losses[f'{i+1}_step_gamma_loss'] = gamma_loss.mean()

      with torch.no_grad():
        mask_changes = dones.float().nonzero().squeeze()
        loss_mask.scatter_(0, mask_changes, 0)
      # TODO: Add stochasticity to next state choice
      encodings = next_logits_pred.argmax(dim=2).detach()

    return losses

  def train(self, batch_data, n=1):
    loss_dict = self.calculate_losses(batch_data, n)
    loss = torch.sum(torch.stack(tuple(loss_dict.values())))

    if self.log_freq > 0 and self.train_step % self.log_freq == 0:
      log_str = f'TransformerTransModel train step {self.train_step} | Loss: {loss.item():.4f}'
      for loss_name, loss_value in loss_dict.items():
        log_str += f' | {loss_name}: {loss_value.item():.4f}'
      print(log_str)

    self.optimizer.zero_grad()
    loss.backward()
    if self.grad_clip > 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    self.optimizer.step()
    self.train_step += 1

    return loss_dict