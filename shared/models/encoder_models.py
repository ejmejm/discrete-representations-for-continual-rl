import copy
from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import *
from .base_structure import FTA


def freeze_model(model, set_eval=True):
  for param in model.parameters():
    param.requires_grad = False
  for buffer in model.buffers():
    buffer.requires_grad = False
  if set_eval:
    model.eval()

def create_encoder(obs_dim, last_fta=False):
  if len(obs_dim) == 1:
    layers = create_dense_layers(obs_dim[0])
  else:
    n_channels = obs_dim[0]
    if obs_dim[1] <= 24:
      layers = create_gridworld_layers(n_channels)
    else:
      layers = create_impala_conv_layers(n_channels)

  if last_fta:
    with torch.no_grad():
      test_data = torch.ones([1] + list(obs_dim), dtype=torch.float32)
      for layer in layers:
        test_data = layer(test_data)
    
    out_shape = test_data.shape
    layers.append(FTA(np.prod(out_shape)))
  else:
    layers.append(nn.ReLU())

  return nn.Sequential(*layers)

def create_decoder(obs_dim):
  if len(obs_dim) == 1:
    layers = create_dense_layers(128, obs_dim[0], [256, 256])
  else:
    n_channels = obs_dim[0]
    if obs_dim[1] <= 24:
      layers = create_gridworld_decoder_layers(n_channels)
    else:
      layers = create_impala_decoder_layers(n_channels)
  return nn.Sequential(*layers)


class DAEModel(nn.Module):
  def __init__(self, obs_dim, encoder=None, decoder=None):
    super().__init__()
    self.encoder_type = 'dae'
    self.encoder = encoder if encoder else create_encoder(obs_dim)
    
    test_input = torch.ones([1] + list(obs_dim), dtype=torch.float32)
    test_encoder_out = self.encoder(test_input)
    self.encoder_out_shape = test_encoder_out.shape[1:]
    self.n_channels = self.encoder_out_shape[0]
    self.n_latent_embeds = np.prod(self.encoder_out_shape[1:])
    self.codebook_size = self.embedding_dim = self.n_channels
    self.conv = nn.Conv2d(self.n_channels, self.n_channels, 1)
    self.decoder = decoder if decoder else create_decoder(obs_dim)

  def encode(self, x, **kwargs):
    """
    Input obs batch (b, c, w, h) -> 
    Output one hot latents batch (b, latent_dim, n_classes)
    """
    encoder_out = self.encoder(x)
    logits = rearrange(encoder_out, 'b c w h -> b (w h) c')
    # I think this is outdated, transpose dims 1 and 2 if I need to use again
    one_hot_latents = logits_to_one_hot(logits, stochastic=False)
    return one_hot_latents

  def decode(self, z):
    """
    Input one hot latents batch (b, latent_dim, n_classes) -> 
    Output obs batch (b, c, w, h)
    """
    disc_one_hot = z.permute(0, 2, 1).reshape(
      z.shape[0], *self.encoder_out_shape).float()
    z = self.conv(disc_one_hot)
    x_hat = self.decoder(disc_one_hot)
    return x_hat

  def forward(self, x):
    latents = self.encoder(x)
    x_hat = self.decoder(latents)
    return x_hat

# prev_vals = []

class AEModel(nn.Module):
  def __init__(self, obs_dim, latent_dim=64, encoder=None, decoder=None,
               stochastic=False, fta=False, fta_params=None,
               latent_activation=False): # Stochastic makes this a VAE
    super().__init__()
    if stochastic:
      self.encoder_type = 'vae'
    elif fta:
      self.encoder_type = 'fta_ae'
    else:
      self.encoder_type = 'ae'
      
    self.stochastic = stochastic
    self.encoder = encoder if encoder else create_encoder(obs_dim)
    
    test_input = torch.ones([1] + list(obs_dim), dtype=torch.float32)
    test_encoder_out = self.encoder(test_input)
    self.encoder_out_shape = test_encoder_out.shape[1:]
    encoder_out_dim = np.prod(self.encoder_out_shape)
    if latent_dim is None:
      self.latent_dim = encoder_out_dim
      self.require_upscale = False
    else:
      self.latent_dim = latent_dim
      self.require_upscale = True

    self.fc_layer = nn.Linear(encoder_out_dim, self.latent_dim)
    
    if fta:
      fta_params = fta_params or {}
      self.fc_activation = FTA(self.latent_dim, **fta_params)
      self.latent_dim *= self.fc_activation.n_tiles
    elif latent_activation:
      self.fc_activation = nn.ReLU()
    else:
      self.fc_activation = nn.Identity()

    if self.stochastic:
      if fta:
        raise ValueError('AEModel cannot be stochastic and use FTA!')
      self.std_layer = nn.Linear(encoder_out_dim, self.latent_dim)
    

    self.decoder = decoder if decoder else create_decoder(obs_dim)
    if self.require_upscale:
      self.z_upscale_layer = nn.Linear(self.latent_dim, encoder_out_dim)
      self.decoder = nn.Sequential(
        self.z_upscale_layer,
        nn.ReLU(),
        ReshapeLayer(-1, *self.encoder_out_shape),
        self.decoder)
    else:
      self.decoder = nn.Sequential(
        ReshapeLayer(-1, *self.encoder_out_shape),
        self.decoder)

  # Currently only works for deterministic encoder (not VAE)
  def get_encoder(self):
    encoder = nn.Sequential(
      self.encoder,
      ReshapeLayer(-1, np.prod(self.encoder_out_shape)),
      self.fc_layer,
      self.fc_activation)

    encoder.encoder_type = self.encoder_type
    encoder.encoder_out_shape = self.encoder_out_shape
    encoder.latent_dim = self.latent_dim

    return encoder

  def encode(self, x, return_all=False, **kwargs):
    encoder_out = self.encoder(x)
    encoder_out = encoder_out.reshape(encoder_out.shape[0], -1)
    means = self.fc_layer(encoder_out)


    # if self.encoder_type == 'fta_ae':
    #   global prev_vals
    #   prev_vals.append(means.detach().cpu())

    #   if len(prev_vals) > 500:
    #     pvs = torch.cat(prev_vals, dim=0).reshape(-1)
    #     # Print percent of values between -2 and 2 to 2 decimal places
    #     print(f'Percent between -2.5 and 2.5: {((pvs > -2.5) & (pvs < 2.5)).float().mean().item():.2f}')

    #     # Now look for each element how frequently it is between -2.5 and 2.5
    #     # Then print the 5th and 95th percentiles (of frequencies) for all elements 
    #     pvs = torch.cat(prev_vals, dim=0)
    #     pvs = pvs.float().flatten()
    #     print(f'5th percentile: {pvs.kthvalue(int(0.05 * pvs.shape[0])).values.item():.2f}')
    #     print(f'25th percentile: {pvs.kthvalue(int(0.25 * pvs.shape[0])).values.item():.2f}')
    #     print(f'75th percentile: {pvs.kthvalue(int(0.75 * pvs.shape[0])).values.item():.2f}')
    #     print(f'95th percentile: {pvs.kthvalue(int(0.95 * pvs.shape[0])).values.item():.2f}')

    #     prev_vals = []


    means = self.fc_activation(means)
    if not self.stochastic:
      if return_all:
        return means, means, None
      return means

    stds = self.std_layer(encoder_out)
    latents = means + stds * torch.normal(
      torch.zeros_like(stds), torch.ones_like(stds))

    if return_all:
      return latents, means, stds
    return latents

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x, return_all=False):
    z, means, stds = self.encode(x, return_all=True)
    x_hat = self.decoder(z)
    if return_all:
      return x_hat, means, stds
    return x_hat

class FlattenModel(nn.Module):
  def __init__(self, obs_dim):
    super().__init__()
    self.encoder_type = 'flatten'
    self.obs_dim = obs_dim
    self.encoder_out_shape = (np.prod(obs_dim),)
    self.latent_dim = self.encoder_out_shape[0]
    self.flatten_layer = ReshapeLayer(-1, self.latent_dim)
    # Just used to keep track of the device, value never changes
    self.device_param = nn.parameter.Parameter(
      torch.tensor(0, dtype=torch.float32))

  def get_encoder(self):
    return self

  def encode(self, x, **kwargs):
    return self.flatten_layer(x)

  def decode(self, z):
    return z.reshape(-1, *self.encoder_out_shape)

  def forward(self, x):
    return self.flatten_layer(x)


class IdentityModel(nn.Module):
  """ Currently used for handcrafted discrete observation spaces. """

  def __init__(self, obs_dim, **kwargs):
    super().__init__()
    self.encoder_type = 'identity'
    self.obs_dim = obs_dim
    self.encoder_out_shape = obs_dim

    for k, v in kwargs.items():
      setattr(self, k, v)

    if len(obs_dim) == 2:
      self.is_one_hot = True
      self.codebook_size = obs_dim[0]
      self.n_embeddings = obs_dim[0]
      
      self.n_latent_embeds = obs_dim[1]

      # Flat representation size
      self.latent_dim = self.n_embeddings * self.n_latent_embeds
    else:
      self.is_one_hot = False
      self.latent_dim = np.prod(obs_dim)

    # Just used to keep track of the device, value never changes
    self.device_param = nn.parameter.Parameter(
      torch.tensor(0, dtype=torch.float32))

  def get_encoder(self):
    return self

  def encode(self, x, **kwargs):
    if self.is_one_hot:
      return torch.argmax(x, dim=1)
    return x

  def decode(self, z):
    if self.is_one_hot:
      one_hots = F.one_hot(z, self.n_embeddings).float()
      return rearrange(one_hots, 'b n c -> b c n')
    return z

  def forward(self, x):
    return x

# Source: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        # Size of the codebook
        self._num_embeddings = n_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(n_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def quantized_decode(self, oh_encodings):
      """ Decodes from one-hot encodings. """
      batch_size, n_latents, latent_dim = oh_encodings.shape
      flat_oh_encodings = oh_encodings.reshape(-1, latent_dim)
      # Quantize and unflatten
      quantized = torch.matmul(flat_oh_encodings, self._embedding.weight.detach())
      quantized = quantized.view(batch_size, n_latents, self._embedding_dim)
      return quantized.permute(0, 2, 1)
      
    def decode(self, encoding_indices):
      """ Decodes from `LongTensor` encodings. """
      batch_size, n_latents = encoding_indices.shape
      encoding_indices = encoding_indices.reshape(-1, 1)
      encodings = F.one_hot(encoding_indices, self._num_embeddings).float()
      # Quantize and unflatten
      quantized = torch.matmul(encodings, self._embedding.weight)
      quantized = quantized.view(batch_size, n_latents, self._embedding_dim)
      return quantized.permute(0, 2, 1)

    def forward(self, inputs, codebook_mask=None, masked_input=True, masked_output=True):
      """
      Input can be BC(WH) or BCWH, C is embedding dim, (WH) is n latents
      convert inputs from BCHW -> BHWC

      codebook_mask only masks codebook for quantization output unless
      masked_input is True, in which case it masks the input as well.
      """

      if len(inputs.shape) == 4:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
      else:
        inputs = inputs.permute(0, 2, 1).contiguous()
      input_shape = inputs.shape
      
      # Flatten input
      flat_input = inputs.view(-1, self._embedding_dim)
      
      # Get the codebook weights
      codebook = self._embedding.weight
      if codebook_mask is not None and masked_input:
        codebook = codebook * codebook_mask
      
      # Calculate distances
      distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                  + torch.sum(codebook**2, dim=1)
                  - 2 * torch.matmul(flat_input, codebook.t()))
      
      # Encoding
      flat_oh_encodings = max_one_hot(-distances)
      oh_encodings = rearrange(flat_oh_encodings,
        '(b n) d -> b d n', b=inputs.shape[0])
      
      # Quantize and unflatten
      with torch.no_grad():
        codebook = self._embedding.weight
        if codebook_mask is not None and masked_output:
          codebook = codebook * codebook_mask
        quantized = torch.matmul(flat_oh_encodings, codebook).view(input_shape)
      
      # Use EMA to update the embedding vectors
      if self.training:
          self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * torch.sum(flat_oh_encodings.detach(), 0)
          
          # Laplace smoothing of the cluster size
          n = torch.sum(self._ema_cluster_size.data)
          self._ema_cluster_size = (
              (self._ema_cluster_size + self._epsilon)
              / (n + self._num_embeddings * self._epsilon) * n)
          
          dw = torch.matmul(flat_oh_encodings.detach().t(), flat_input)
          self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
          
          self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
      
      # Loss
      e_latent_loss = F.mse_loss(quantized, inputs)
      loss = self._commitment_cost * e_latent_loss
      
      # Straight Through Estimator
      quantized = inputs + (quantized - inputs).detach()
      avg_probs = torch.mean(flat_oh_encodings, dim=0)
      perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
      
      # convert quantized from BHWC -> BCHW
      if len(input_shape) == 4:
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, oh_encodings
      return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, oh_encodings

class VQVAEModel(nn.Module):
  def __init__(self, obs_dim, codebook_size, embedding_dim, encoder=None,
               decoder=None, n_latents=None, quantized_enc=False, sparsity=0.0,
               sparsity_type='random'):
    super().__init__()
    if quantized_enc:
      self.encoder_type = 'soft_vqvae'
    else:
      self.encoder_type = 'vqvae'
    self.codebook_size = codebook_size
    self.embedding_dim = embedding_dim
    self.n_embeddings = embedding_dim if quantized_enc else codebook_size
    self.quantized_enc = quantized_enc
    self.encoder = encoder or create_encoder(obs_dim)
    self.quantizer = VectorQuantizerEMA(
      codebook_size, embedding_dim,
      commitment_cost=0.25, decay=0.99)
    self.decoder = decoder or create_decoder(obs_dim)

    test_input = torch.ones(1, *obs_dim, dtype=torch.float32)
    self.encoder_out_shape = self.encoder(test_input).shape[1:]
    self.n_latent_embeds = np.prod(self.encoder_out_shape[1:])
    
    if n_latents is not None:
      if len(self.encoder_out_shape) == 1:
        # When the encoder has a flat output, make a layer to create channels
        self.encoder = nn.Sequential(
          self.encoder,
          nn.Linear(self.encoder_out_shape[0], embedding_dim * n_latents),
          ReshapeLayer(-1, embedding_dim, n_latents))
        self.decoder = nn.Sequential(
          ReshapeLayer(-1, embedding_dim * n_latents),
          nn.Linear(embedding_dim * n_latents, self.encoder_out_shape[0]),
          self.decoder)
        
        self.encoder_out_shape = (embedding_dim, n_latents)
        self.n_latent_embeds = n_latents
      else:
        raise NotImplementedError('`n_latents` param not supported for >1-D encoder outputs!')
    elif len(self.encoder_out_shape) == 1:
      raise ValueError('VQVAEs with dense encoders must have a value for `n_latents`!')

    # Create sparsity masks if needed
    sparsity_mask = self.create_sparsity_mask(sparsity, sparsity_type)
    if sparsity_mask is None:
      self.sparsity_mask = None
    else:
      self.register_buffer('sparsity_mask', sparsity_mask)
    self.sparsity_enabled = False

    # Flat representation size
    # num classes x num vectors in a single latent vector
    self.latent_dim = self.n_embeddings * self.n_latent_embeds

  def get_codebook(self):
    codebook = self.quantizer._embedding.weight
    if self.sparsity_enabled and self.sparsity_mask is not None:
      codebook = codebook * self.sparsity_mask
    # (n_embeddings, embedding_dim)
    return codebook

  def create_sparsity_mask(self, sparsity=0.0, sparsity_type='random'):
    if self.encoder_type == 'soft_vqvae' and \
       (sparsity > 0 or sparsity_type == 'identity'):
      if sparsity_type == 'random':
        random_idxs = torch.randn((self.n_embeddings, self.embedding_dim))
        sorted_indices = torch.sort(random_idxs, dim=1)[1]
        n_zeros = int(self.embedding_dim * sparsity)
        sparsity_mask = sorted_indices >= n_zeros
        sparsity_mask = sparsity_mask.float()
      elif sparsity_type == 'identity':
        assert self.n_embeddings == self.embedding_dim, \
          'Identity sparsity only supported for square matrices!'
        sparsity_mask = torch.eye(self.embedding_dim)
    else:
      sparsity_mask = None
    
    return sparsity_mask

  def enable_sparsity(self):
    self.sparsity_enabled = True

  def disable_sparsity(self):
    self.sparsity_enabled = False

  def get_encoder(self):
    new_quantizer = copy.copy(self.quantizer)
    new_quantizer.full_forward = self.quantizer.forward
    if self.quantized_enc:
      new_quantizer.forward = lambda x: new_quantizer.full_forward(x)[1]
    else:
      new_quantizer.forward = lambda x: new_quantizer.full_forward(x)[3]

    encoder = nn.Sequential(
      self.encoder,
      new_quantizer
    )

    encoder.encoder_type = self.encoder_type
    encoder.encoder_out_shape = self.encoder_out_shape
    encoder.n_latent_embeds = self.n_latent_embeds
    encoder.n_embeddings = self.n_embeddings
    encoder.embedding_dim = self.embedding_dim

    return encoder

  def forward(self, x):
    encoder_out = self.encoder(x)
    if self.sparsity_enabled and self.sparsity_mask is not None:
      quantizer_loss, quantized, perplexity, oh_encodings = self.quantizer(
        encoder_out, self.sparsity_mask, masked_input=False, masked_output=False)
      x_hat = self.decoder(quantized)
      quantized = self.quantizer(encoder_out, self.sparsity_mask, masked_input=False)[1]
    else:
      quantizer_loss, quantized, perplexity, oh_encodings = self.quantizer(encoder_out)
      x_hat = self.decoder(quantized)
    out_encodings = quantized.reshape(x.shape[0], self.n_embeddings, -1) \
      if self.quantized_enc else oh_encodings

    return x_hat, quantizer_loss, perplexity, out_encodings

  def encode(self, x, return_one_hot=False, as_long=True):
    encoder_out = self.encoder(x)
    mask = self.sparsity_mask if self.sparsity_enabled else None
    _, quantized, quantizer_loss, oh_encodings = self.quantizer(
      encoder_out, mask, masked_input=False)
    if self.quantized_enc:
      quantized = quantized.reshape(x.shape[0], self.n_embeddings, -1)
      return quantized
    elif return_one_hot or not as_long:
      return oh_encodings
    else:
      return oh_encodings.argmax(dim=1)

  def decode(self, encodings):
    if self.quantized_enc:
      quantized = encodings
      if self.sparsity_enabled and self.sparsity_mask is not None:
        quantized = quantized.view(
          quantized.shape[0], self.n_embeddings, self.n_latent_embeds)
        quantized = self.quantizer(
          quantized, self.sparsity_mask, masked_output=False)[1]
    else:
      quantized = self.quantizer.decode(encodings.long())
    quantized = quantized.view(encodings.shape[0], *self.encoder_out_shape)
    return self.decoder(quantized)

  def quantize_logits(self, logits):
    quantizer_loss, quantized = self.quantizer(logits)[:2]
    return quantized, quantizer_loss

  def decode_from_quantized(self, quantized):
    quantized = quantized.view(quantized.shape[0], *self.encoder_out_shape)
    return self.decoder(quantized)

class SoftmaxAEModel(nn.Module):
  """
  n_embeddings == codebook_size, which is the number of classes.
  The filter size determines the number of latent embeddings.
  """

  def __init__(
      self, obs_dim, codebook_size, encoder=None, decoder=None,
      n_latents=None, unimix=0.01):
    super().__init__()
    self.encoder_type = 'softmax_ae'
    self.unimix = unimix
    self.n_embeddings = self.cookbook_size = codebook_size
    self.encoder = encoder or create_encoder(obs_dim)
    self.decoder = decoder or create_decoder(obs_dim)

    test_input = torch.ones(1, *obs_dim, dtype=torch.float32)
    self.encoder_out_shape = self.encoder(test_input).shape[1:]
    self.embedding_dim = self.encoder_out_shape[0]
    self.n_latent_embeds = np.prod(self.encoder_out_shape[1:])
    
    if n_latents is not None:
      if len(self.encoder_out_shape) == 1:
        # When the encoder has a flat output, make a layer to create channels
        self.encoder = nn.Sequential(
          self.encoder,
          nn.Linear(self.embedding_dim, self.n_embeddings * n_latents),
          ReshapeLayer(-1, self.n_embeddings, n_latents))
        self.decoder = nn.Sequential(
          ReshapeLayer(-1, self.n_embeddings * n_latents),
          nn.Linear(self.n_embeddings * n_latents, self.embedding_dim),
          self.decoder)
        
        self.encoder_out_shape = (self.n_embeddings, n_latents)
        self.n_latent_embeds = n_latents
      else:
        raise NotImplementedError('`n_latents` param not supported for >1-D encoder outputs!')
    elif len(self.encoder_out_shape) == 1:
      raise ValueError('Discrete models with dense encoders must have a value for `n_latents`!')
    
    if len(self.encoder_out_shape) == 3:
      self.encoder = nn.Sequential(
        self.encoder,
        nn.Conv2d(self.embedding_dim, self.n_embeddings, kernel_size=1, stride=1))
      # Equivalent to adding embeddings
      self.decoder = nn.Sequential(
        nn.Conv2d(self.n_embeddings, self.embedding_dim, kernel_size=1, stride=1),
        self.decoder)
      
    self.latent_dim = self.n_embeddings * self.n_latent_embeds

  def encode(self, x: torch.FloatTensor, return_one_hot=False, as_long=True) -> torch.LongTensor:
    encoder_out = self.encoder(x)
    
    # encoder_out: batch size, classes, filter height, filter width
    # Merge filter dims
    logits = encoder_out.reshape(*encoder_out.shape[:2], -1)

    # If unimix, mix in uniform distribution to one-hot probs
    if self.unimix > 0:
      probs = F.softmax(encoder_out, dim=1)
      uniform_distrib = torch.ones_like(encoder_out) / encoder_out.shape[1]
      probs = (1 - self.unimix) * probs + self.unimix * uniform_distrib
      logits = torch.log(probs)

    # Sample one-hots from the logits
    one_hot_samples = sample_one_hot(logits, dim=1)

    if return_one_hot or not as_long:
      return one_hot_samples
    
    discrete_z = one_hot_samples.argmax(dim=1).reshape(x.shape[0], -1)
    return discrete_z

  def decode(self, encodings: torch.LongTensor) -> torch.FloatTensor:
    oh_encodings = F.one_hot(encodings, num_classes=self.n_embeddings).float()
    oh_encodings = rearrange(oh_encodings, 'b n c -> b c n')
    oh_encodings = oh_encodings.reshape(-1, self.n_embeddings, *self.encoder_out_shape[1:])
    return self.decoder(oh_encodings)
  
  def forward(self, x):
    encoder_out = self.encoder(x)
    # encoder_out: batch size, classes, filter height, filter width
    # Merge filter dims
    logits = encoder_out.reshape(*encoder_out.shape[:2], -1)

    # If unimix, mix in uniform distribution to one-hot probs
    if self.unimix > 0:
      probs = F.softmax(encoder_out, dim=1)
      uniform_distrib = torch.ones_like(encoder_out) / encoder_out.shape[1]
      probs = (1 - self.unimix) * probs + self.unimix * uniform_distrib
      logits = torch.log(probs)

    # Sample one-hots from the logits
    one_hot_samples = sample_one_hot(logits, dim=1)

    # Decode
    x_hat = self.decoder(one_hot_samples)

    return x_hat
  
class PermuteLayer(nn.Module):
  def __init__(self, *dims):
    self.dims = dims
    
  def forward(self, x):
    return x.permute(*self.dims)
  
class HardFTAAEModel(nn.Module):
  def __init__(self, obs_dim, codebook_size, encoder=None, decoder=None, n_latents=None):
    super().__init__()
    self.encoder_type = 'hard_fta_ae'
    self.n_embeddings = codebook_size
    self.encoder = encoder or create_encoder(obs_dim)
    self.decoder = decoder or create_decoder(obs_dim)

    test_input = torch.ones(1, *obs_dim, dtype=torch.float32)
    self.encoder_out_shape = self.encoder(test_input).shape[1:]
    self.embedding_dim = self.encoder_out_shape[0]
    self.n_latent_embeds = np.prod(self.encoder_out_shape[1:])
    
    if len(self.encoder_out_shape) == 1:
      # When the encoder has a flat output, make a layer to create channels
      if n_latents is not None:
        self.encoder = nn.Sequential(
          self.encoder,
          nn.Linear(self.encoder_out_shape[0], n_latents))
        self.decoder = nn.Sequential(
          nn.Linear(n_latents, self.encoder_out_shape[0]),
          self.decoder)
        
        self.encoder_out_shape = (n_latents,)
        self.n_latent_embeds = n_latents
      else:
        raise ValueError('`n_latents` is required for 1-D encoder outputs!')
    elif len(self.encoder_out_shape) == 3:
      if n_latents is not None:
        raise NotImplementedError('`n_latents` param not supported for >1-D encoder outputs!')
      else:
        self.encoder = nn.Sequential(
          self.encoder,
          nn.Conv2d(self.encoder_out_shape[0], 1, kernel_size=1, stride=1, padding=0),
          ReshapeLayer(-1, self.n_latent_embeds),
          FTA(self.n_latent_embeds, tiles=self.n_embeddings, eta=0),
          ReshapeLayer(-1, *self.encoder_out_shape[1:], self.n_embeddings),
        )
        self.decoder = nn.Sequential(
          nn.Conv2d(self.n_embeddings, self.encoder_out_shape[0], kernel_size=1, stride=1, padding=0),
          self.decoder)
        self.encoder_out_shape = (self.n_embeddings, *self.encoder_out_shape[1:])
    else:
      raise ValueError(f'Encoder output of shape {self.encoder_out_shape} is not supported!')

  def encode(self, x: torch.FloatTensor, as_long=True) -> torch.LongTensor:
    encoder_out = self.encoder(x)
    encoder_out = encoder_out.permute(0, 3, 1, 2)
    if not as_long:
      return encoder_out.reshape(*encoder_out.shape[:2], -1)
    discrete_z = encoder_out.argmax(dim=1).reshape(x.shape[0], -1)
    return discrete_z

  def decode(self, encodings: torch.LongTensor) -> torch.FloatTensor:
    oh_encodings = F.one_hot(encodings, num_classes=self.n_embeddings).float()
    oh_encodings = rearrange(oh_encodings, 'b n c -> b c n')
    oh_encodings = oh_encodings.reshape(oh_encodings.shape[0], *self.encoder_out_shape)
    return self.decoder(oh_encodings)
  
  def forward(self, x):
    # One-hot output
    encoder_out = self.encoder(x)
    encoder_out = encoder_out.permute(0, 3, 1, 2)
    x_hat = self.decoder(encoder_out)
    return x_hat