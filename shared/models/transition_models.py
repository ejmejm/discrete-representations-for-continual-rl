import math

from einops import rearrange
from gym import spaces
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import max_one_hot, mlp, sample_one_hot


class DiscreteTransitionModel(nn.Module):
  def __init__(self, input_dim, n_embeddings, embedding_dim, act_space,
      hidden_sizes=[256, 256], state_head_size=None, reward_head_size=64,
      gamma_head_size=64, logits_to_state_func=None, stochastic=None,
      n_trans_options=32, stoch_hidden_sizes=[128], discretizer_hidden_sizes=[],
      use_soft_embeds=False, return_logits=False):
    super().__init__()
    self.input_dim = input_dim
    self.embedding_dim = embedding_dim
    self.n_embeddings = n_embeddings
    # When true a 1D Conv is used to embed the state
    # This can work with not truly discrete states
    self.use_soft_embeds = use_soft_embeds
    self.return_logits = return_logits
    if isinstance(act_space, spaces.Discrete):
      self.act_dim = act_space.n
      self.act_dtype = torch.long
    else:
      self.act_dim = np.prod(act_space.shape)
      self.act_dtype = torch.float32
    self.cb_flat_dim = input_dim * embedding_dim
    if logits_to_state_func:
      self.logits_to_state = logits_to_state_func
    self.stochastic = stochastic
    if self.stochastic == 'categorical':
      self.n_trans_options = n_trans_options

      stoch_hidden_sizes = [self.cb_flat_dim + self.act_dim] \
        + stoch_hidden_sizes + [self.n_trans_options]
      self.stoch_proj = mlp(stoch_hidden_sizes)

      discretizer_hidden_sizes = [self.cb_flat_dim] \
        + discretizer_hidden_sizes + [self.n_trans_options]
      self.disc_proj = mlp(discretizer_hidden_sizes)
    else:
      self.n_trans_options = 0

    self.state_head_size = state_head_size or hidden_sizes[-1]

    if self.use_soft_embeds:
      assert n_embeddings == embedding_dim
      # TODO: Consider changing this to Identity
      self.embeddings = nn.Identity() # nn.Conv1d(n_embeddings, embedding_dim, 1)
    else:
      self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
    
    hidden_sizes = [self.cb_flat_dim + self.act_dim + self.n_trans_options] + hidden_sizes
    self.shared_layers = mlp(hidden_sizes)

    self.state_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], self.state_head_size),
      nn.ReLU(),
      nn.Linear(self.state_head_size, n_embeddings * input_dim))

    self.reward_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], reward_head_size),
      nn.ReLU(),
      nn.Linear(reward_head_size, 1))

    self.gamma_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], gamma_head_size),
      nn.ReLU(),
      nn.Linear(gamma_head_size, 1))

  def discretize(self, x: torch.FloatTensor, return_logits: bool = False):
    """
    Given a continuous state returns a one hot
    encoding of the discretized state.
    """
    return_logits = return_logits or self.return_logits

    if self.stochastic != 'categorical':
      raise ValueError('Can only discretize if stochasticity type is categorical!')
    input_embeds = self.embeddings(x).view(x.shape[0], -1)
    logits = self.disc_proj(input_embeds)
    oh_states = max_one_hot(logits.softmax(dim=-1))
    # oh_states = F.gumbel_softmax(logits, hard=True, dim=1)
    if return_logits:
      return oh_states, logits
    return oh_states

  def logits_to_state(self, logits: torch.FloatTensor, dim: int = 1,
                      one_hot: bool = False):
    states = sample_one_hot(logits, dim=dim)
    if one_hot:
      return states
    return states.argmax(dim=dim)

  def prepare_acts(self, acts):
    if self.act_dtype == torch.long:
      acts = F.one_hot(acts, self.act_dim).float()
    return acts
  
  def forward(self, x: torch.Tensor, acts: torch.Tensor,
              oh_outcomes=None, return_one_hots=False,
              return_logits=False, return_stoch_logits=False):
    # x could be longs for true discrete states or floats
    # when using soft embeddings
    return_one_hots = return_one_hots or self.use_soft_embeds
    return_logits = return_logits or self.return_logits

    embeds = self.embeddings(x)
    flat_embeds = embeds.view(embeds.shape[0], -1)
    processed_acts = self.prepare_acts(acts)
    input_embeds = torch.cat([flat_embeds, processed_acts], dim=1)

    if self.stochastic == 'categorical':
      stoch_logits = self.stoch_proj(input_embeds) # sigma logits
      if oh_outcomes is None:
        stoch_probs = F.softmax(stoch_logits, dim=1) # sigma
        samples = torch.multinomial(stoch_probs, 1) # sample
        samples = samples.reshape(*stoch_probs.shape[:-1])
        oh_outcomes = F.one_hot(samples, num_classes=self.n_trans_options)
      input_embeds = torch.cat([input_embeds, oh_outcomes], dim=1)
    else:
      stoch_logits = None

    z = self.shared_layers(input_embeds)

    state_logits = self.state_head(z)
    state_logits = state_logits.reshape(x.shape[0], self.n_embeddings, self.input_dim)

    reward = self.reward_head(z)

    gamma = self.gamma_head(z)
    gamma = torch.sigmoid(gamma)

    if return_logits:
      out_states = state_logits
    else:
      out_states = self.logits_to_state(
        state_logits, one_hot=return_one_hots)

    if return_stoch_logits:
      return out_states, reward, gamma, stoch_logits
    return out_states, reward, gamma

class ContinuousTransitionModel(nn.Module):
  def __init__(self, input_dim, act_space, hidden_sizes=[256, 256],
      state_head_size=256, reward_head_size=64, gamma_head_size=64,
      stochastic=None, n_trans_options=32, stoch_hidden_sizes=[128],
      discretizer_hidden_sizes=[], logits_to_state_func=None):
    """
    stochastic options: [None|'categorical'|'normal']
    """
    super().__init__()

    if logits_to_state_func is not None:
      self.logits_to_state = logits_to_state_func

    self.input_dim = input_dim
    if isinstance(act_space, spaces.Discrete):
      self.act_dim = act_space.n
      self.act_dtype = torch.long
    else:
      self.act_dim = np.prod(act_space.shape)
      self.act_dtype = torch.float32
    self.stochastic = stochastic
    if self.stochastic == 'categorical':
      self.n_trans_options = n_trans_options

      stoch_hidden_sizes = [self.input_dim + self.act_dim] \
        + stoch_hidden_sizes + [self.n_trans_options]
      self.stoch_proj = mlp(stoch_hidden_sizes)

      discretizer_hidden_sizes = [self.input_dim] \
        + discretizer_hidden_sizes + [self.n_trans_options]
      self.disc_proj = mlp(discretizer_hidden_sizes)
    else:
      self.n_trans_options = 0

    hidden_sizes = [self.input_dim + self.act_dim + self.n_trans_options] + hidden_sizes
    self.shared_layers = mlp(hidden_sizes)

    self.state_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], state_head_size),
      nn.ReLU(),
      nn.Linear(state_head_size, self.input_dim))

    self.reward_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], reward_head_size),
      nn.ReLU(),
      nn.Linear(reward_head_size, 1))

    self.gamma_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], gamma_head_size),
      nn.ReLU(),
      nn.Linear(gamma_head_size, 1))

  def logits_to_state(self, x):
    return x

  def discretize(self, x: torch.FloatTensor, return_logits: bool = False):
    """
    Given a continuous state returns a one hot
    encoding of the discretized state.
    """
    x = x.view(x.shape[0], self.input_dim)
    if self.stochastic != 'categorical':
      raise ValueError('Can only discretize if stochasticity type is categorical!')
    logits = self.disc_proj(x)
    oh_states = max_one_hot(logits.softmax(dim=-1))
    # oh_states = F.gumbel_softmax(logits, hard=True, dim=1)
    if return_logits:
      return oh_states, logits
    return oh_states

  def prepare_acts(self, acts: torch.Tensor):
    if self.act_dtype == torch.long:
      acts = F.one_hot(acts, self.act_dim).float()
    return acts
      
  def forward(self, x: torch.FloatTensor, acts: torch.Tensor,
              oh_outcomes=None, return_logits=False,
              return_stoch_logits: bool = False):
    x = x.view(x.shape[0], self.input_dim)
    processed_acts = self.prepare_acts(acts)
    input_embeds = torch.cat([x, processed_acts], dim=1)

    if self.stochastic == 'categorical':
      stoch_logits = self.stoch_proj(input_embeds) # sigma logits
      if oh_outcomes is None:
        stoch_probs = F.softmax(stoch_logits, dim=1) # sigma
        samples = torch.multinomial(stoch_probs, 1) # sample
        samples = samples.reshape(*stoch_probs.shape[:-1])
        oh_outcomes = F.one_hot(samples, num_classes=self.n_trans_options)
      input_embeds = torch.cat([input_embeds, oh_outcomes], dim=1)
    else:
      stoch_logits = None

    z = self.shared_layers(input_embeds)

    states = self.state_head(z)
    if not return_logits:
      states = self.logits_to_state(states)

    reward = self.reward_head(z)
    gamma = self.gamma_head(z)
    gamma = torch.sigmoid(gamma)

    if return_stoch_logits:
      return states, reward, gamma, stoch_logits
    return states, reward, gamma
  

class UniversalVQTransitionModel(nn.Module):
  def __init__(self, input_dim, n_embeddings, embedding_dim, act_space,
      hidden_sizes=[256, 256], state_head_size=None, reward_head_size=64,
      gamma_head_size=64, logits_to_state_func=None, stochastic=None,
      n_trans_options=32, stoch_hidden_sizes=[128], discretizer_hidden_sizes=[],
      use_soft_embeds=False, return_logits=False, use_1d_conv=False,
      embed_snap_encoder=None, **kwargs):
    """
    use_1d_conv: When true a 1D Conv is used to embed the state
      This can work with not truly discrete states

    embed_snap_encoder: A VQVAE encoder, required for embed snap when using soft embeddings
    """
    super().__init__()
    self.input_dim = input_dim
    self.embedding_dim = embedding_dim
    self.n_embeddings = n_embeddings
    # When true a 1D Conv is used to embed the state
    # This can work with not truly discrete states
    self.use_soft_embeds = use_soft_embeds
    self.return_logits = return_logits
    if isinstance(act_space, spaces.Discrete):
      self.act_dim = act_space.n
      self.act_dtype = torch.long
    else:
      self.act_dim = np.prod(act_space.shape)
      self.act_dtype = torch.float32
    self.cb_flat_dim = input_dim * embedding_dim
    if logits_to_state_func:
      self.logits_to_state = logits_to_state_func
    self.stochastic = stochastic
    self.embed_snap_encoder = embed_snap_encoder

    if kwargs.get('rand_mask', False):
      rand_mask = torch.randn(n_embeddings, input_dim)
      self.register_buffer('rand_mask', rand_mask)
    else:
      self.rand_mask = None

    if self.stochastic == 'categorical':
      self.n_trans_options = n_trans_options

      stoch_hidden_sizes = [self.cb_flat_dim + self.act_dim] \
        + stoch_hidden_sizes + [self.n_trans_options]
      self.stoch_proj = mlp(stoch_hidden_sizes)

      discretizer_hidden_sizes = [self.cb_flat_dim] \
        + discretizer_hidden_sizes + [self.n_trans_options]
      self.disc_proj = mlp(discretizer_hidden_sizes)
    else:
      self.n_trans_options = 0

    self.state_head_size = state_head_size or hidden_sizes[-1]

    if use_1d_conv:
      self.embeddings = nn.Conv1d(n_embeddings, embedding_dim, 1, bias=False)
      # Init as normal, same as embedding layer
      nn.init.normal_(self.embeddings.weight)

      # (1, n_embeddings)
      scale_factor = kwargs.get('embed_scale_factor', None)

      if scale_factor is not None:
        assert scale_factor.shape == (1, n_embeddings), \
          'Scale factor must be of shape (1, n_embeddings)'
        # reparametrization trick
        with torch.no_grad():
          device = self.embeddings.weight.device
          self.embeddings.weight.copy_(
            scale_factor.to(device)[..., None] * self.embeddings.weight)

        if kwargs.get('embed_grad_hook', False):
          # Register the gradient scaling hook
          self.embeddings.weight.register_hook(
            lambda grad: grad / scale_factor[..., None])

    else:
      assert n_embeddings == embedding_dim, \
        'Embedding dim must match n_embeddings or use 1d conv'
      self.embeddings = nn.Identity()
    
    hidden_sizes = [self.cb_flat_dim + self.act_dim + self.n_trans_options] + hidden_sizes
    self.shared_layers = mlp(hidden_sizes)

    self.state_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], self.state_head_size),
      nn.ReLU(),
      nn.Linear(self.state_head_size, n_embeddings * input_dim))

    self.reward_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], reward_head_size),
      nn.ReLU(),
      nn.Linear(reward_head_size, 1))

    self.gamma_head = nn.Sequential(
      nn.Linear(hidden_sizes[-1], gamma_head_size),
      nn.ReLU(),
      nn.Linear(gamma_head_size, 1))

  def discretize(self, x: torch.FloatTensor, return_logits: bool = False):
    """
    Given a continuous state returns a one hot
    encoding of the discretized state.
    """
    return_logits = return_logits or self.return_logits

    if self.stochastic != 'categorical':
      raise ValueError('Can only discretize if stochasticity type is categorical!')
    
    if not self.use_soft_embeds:
      # Convert x to one hot
      x = F.one_hot(x, self.n_embeddings).float()
      x = rearrange(x, 'b ... c -> b c ...')

    if self.rand_mask is not None:
      with torch.no_grad():
        x *= self.rand_mask[None]

    input_embeds = self.embeddings(x).reshape(x.shape[0], -1)
    logits = self.disc_proj(input_embeds)
    oh_states = max_one_hot(logits.softmax(dim=-1))
    if return_logits:
      return oh_states, logits
    return oh_states

  def logits_to_state(self, logits: torch.FloatTensor, dim: int = 1,
                      one_hot: bool = False):
    if self.use_soft_embeds and self.embed_snap_encoder is not None:
      mask = self.embed_snap_encoder.sparsity_mask if \
        self.embed_snap_encoder.sparsity_enabled else None
      with torch.no_grad():
        states = self.embed_snap_encoder.quantizer(logits, mask)[1]
    elif self.rand_mask is not None:
      states = logits
      if self.embed_snap_encoder is not None:
        with torch.no_grad():
          diffs = (self.rand_mask[None] - logits) ** 2
          states = diffs.argmin(dim=dim)
          # min_diffs = diffs.argmin(dim=dim)
          # states = F.one_hot(min_diffs, diffs.shape[dim]).float()
          # states = rearrange(states, 'b ... c -> b c ...')
          # states = states * self.rand_mask[None]
    elif not self.use_soft_embeds:
      states = sample_one_hot(logits, dim=dim)
      if not one_hot:
        states = states.argmax(dim=dim)
    else:
      states = logits

    return states

  def prepare_acts(self, acts):
    if self.act_dtype == torch.long:
      acts = F.one_hot(acts, self.act_dim).float()
    return acts
  
  def forward(self, x: torch.Tensor, acts: torch.Tensor,
              oh_outcomes=None, return_one_hots=False,
              return_logits=False, return_stoch_logits=False):
    return_logits = return_logits or self.return_logits

    if not self.use_soft_embeds:
      # Convert x to one hot
      x = F.one_hot(x, self.n_embeddings).float()
      x = rearrange(x, 'b ... c -> b c ...')
    else:
      x = x.reshape(x.shape[0], self.n_embeddings, self.input_dim)

    if self.rand_mask is not None:
      with torch.no_grad():
        x *= self.rand_mask[None]

    embeds = self.embeddings(x)
    flat_embeds = embeds.reshape(embeds.shape[0], -1)
    processed_acts = self.prepare_acts(acts)
    input_embeds = torch.cat([flat_embeds, processed_acts], dim=1)

    if self.stochastic == 'categorical':
      stoch_logits = self.stoch_proj(input_embeds) # sigma logits
      if oh_outcomes is None:
        stoch_probs = F.softmax(stoch_logits, dim=1) # sigma
        samples = torch.multinomial(stoch_probs, 1) # sample
        samples = samples.reshape(*stoch_probs.shape[:-1])
        oh_outcomes = F.one_hot(samples, num_classes=self.n_trans_options)
      input_embeds = torch.cat([input_embeds, oh_outcomes], dim=1)
    else:
      stoch_logits = None

    z = self.shared_layers(input_embeds)

    state_logits = self.state_head(z)
    state_logits = state_logits.reshape(x.shape[0], self.n_embeddings, self.input_dim)

    reward = self.reward_head(z)

    gamma = self.gamma_head(z)
    gamma = torch.sigmoid(gamma)

    if return_logits:
      out_states = state_logits
    else:
      out_states = self.logits_to_state(
        state_logits, one_hot=return_one_hots)

    if return_stoch_logits:
      return out_states, reward, gamma, stoch_logits
    return out_states, reward, gamma

class PositionalEncoding(nn.Module):
  # Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int = 1000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_p)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
    pe = torch.zeros(max_len, 1, dim_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(0, 1)
    self.register_buffer('pos_encoding', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor, shape [batch_size, seq_len, embedding_dim]
    """
    x = x + self.pos_encoding[:, :x.size(1)]
    return self.dropout(x)

# Source: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
class TransformerTransitionModel(nn.Module):
  # Constructor
  def __init__(self, n_embeddings, embedding_dim, act_space, num_heads,
               num_encoder_layers, num_decoder_layers, dim_feedforward,
               dropout, stochastic=None):
    super().__init__()
    # INFO
    self.model_type = 'Transformer'
    self.embedding_dim = embedding_dim
    self.n_embeddings = n_embeddings
    if isinstance(act_space, spaces.Discrete):
      self.act_dim = act_space.n
      self.act_dtype = torch.long
    else:
      self.act_dim = np.prod(act_space.shape)
      self.act_dtype = torch.float32
      raise NotImplementedError('Continuous actions not supported!')
    self.stochastic = stochastic
    self.special_token = self.n_embeddings + self.act_dim

    # LAYERS
    self.positional_encoder = PositionalEncoding(
        dim_model=embedding_dim, dropout_p=dropout, max_len=1024
    )
    self.embedding = nn.Embedding(n_embeddings + self.act_dim + 1, embedding_dim)
    self.transformer = nn.Transformer(
        d_model=embedding_dim,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    self.out = nn.Linear(embedding_dim, n_embeddings)
      
  def prepare_acts(self, acts):
    if self.act_dtype == torch.long:
      acts = F.one_hot(acts, self.act_dim).float()
    return acts
  
  def forward(self, state, acts, tgt=None, tgt_mask=None, src_pad_mask=None,
              tgt_pad_mask=None, return_logits=False):
    # State size must be (batch_size, state sequence length)
    # Tgt size must be (batch_size, tgt sequence length)

    # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
    adjusted_acts = acts + self.n_embeddings
    src = torch.cat([state, adjusted_acts.unsqueeze(1)], dim=1)
    src = self.embedding(src) * math.sqrt(self.embedding_dim)
    src = self.positional_encoder(src)

    output_start_tokens = torch.full(
      (src.shape[0], 1), self.special_token, dtype=torch.long, device=src.device)

    if tgt is None:
      tgt = output_start_tokens
      if return_logits:
        out_logits = torch.zeros((*tgt.shape, self.n_embeddings),
          dtype=torch.float32, device=src.device)
      for _ in range(state.shape[1]):
        step_mask = self.get_tgt_mask(tgt.shape[1]).to(tgt.device)
        step_out, reward, gamma = self.forward(
          state, acts, tgt, tgt_mask=step_mask, return_logits=return_logits)
        next_token = step_out[:, -1:]
        if return_logits:
          out_logits = torch.cat([out_logits, next_token], dim=1)
          if self.stochastic:
            probs = F.softmax(next_token, dim=2)
            flat_probs = probs.view(-1, self.n_embeddings)
            sample = torch.multinomial(flat_probs, 1)
            sample = sample.reshape(*probs.shape[:-1])
            next_token = sample
          else:
            next_token = next_token.argmax(dim=2)
        tgt = torch.cat([tgt, next_token], dim=1)
      if return_logits:
        tgt = out_logits
      return tgt[:, 1:], reward, gamma

    tgt = torch.cat([output_start_tokens, tgt], dim=1)
    tgt = tgt[:, :-1]

    tgt = self.embedding(tgt) * math.sqrt(self.embedding_dim)
    tgt = self.positional_encoder(tgt)
    # Size (sequence length, batch_size, dim_model)

    # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
    transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask,
      src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
    state_logits = self.out(transformer_out)
    if return_logits:
      return state_logits, torch.tensor(0), torch.tensor(1)

    if self.stochastic:
      probs = F.softmax(state_logits, dim=2)
      flat_probs = probs.view(-1, self.n_embeddings)
      samples = torch.multinomial(flat_probs, 1)
      samples = samples.reshape(*probs.shape[:-1])
      return samples, torch.tensor(0), torch.tensor(1)
    return state_logits.argmax(dim=2), torch.tensor(0), torch.tensor(1)
    
  def get_tgt_mask(self, size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    
    return mask
  
  def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    return (matrix == pad_token)

# Source: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
class TransformerDecTransitionModel(nn.Module):
  # Constructor
  def __init__(self, n_embeddings, embedding_dim, act_space, num_heads,
               num_decoder_layers, dim_feedforward, dropout, stochastic=None):
    super().__init__()
    # INFO
    self.model_type = 'TransformerDec'
    self.embedding_dim = embedding_dim
    self.n_embeddings = n_embeddings
    if isinstance(act_space, spaces.Discrete):
      self.act_dim = act_space.n
      self.act_dtype = torch.long
    else:
      self.act_dim = np.prod(act_space.shape)
      self.act_dtype = torch.float32
      raise NotImplementedError('Continuous actions not supported!')
    self.stochastic = stochastic
    self.special_token = self.n_embeddings + self.act_dim

    # LAYERS
    self.positional_encoder = PositionalEncoding(
        dim_model=embedding_dim, dropout_p=dropout, max_len=1024
    )
    self.embedding = nn.Embedding(n_embeddings + self.act_dim + 1, embedding_dim)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=embedding_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout)
    self.transformer = nn.TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_decoder_layers
    )
    self.out = nn.Linear(embedding_dim, n_embeddings)
      
  def logits_to_state(self, logits: torch.FloatTensor):
    if self.stochastic in ('simple', 'categorical'):
      return sample_one_hot(logits)
    return max_one_hot(logits)

  def _inference_forward(self, state, acts, return_logits=False):
    adjusted_acts = acts + self.n_embeddings
    src = torch.cat([state, adjusted_acts.unsqueeze(1)], dim=1)
    src_act_size = src.shape[1]

    output_start_tokens = torch.full(
      (src.shape[0], 1), self.special_token, dtype=torch.long, device=src.device)
    
    tgt = output_start_tokens
    out_logits = []
    while tgt.shape[1] <= state.shape[1]:
      full_seqs = torch.cat([src, tgt], dim=1)

      full_seqs = self.embedding(full_seqs) * math.sqrt(self.embedding_dim)
      full_seqs = self.positional_encoder(full_seqs)
      full_seqs = full_seqs.permute(1, 0, 2)
      # Size (batch_size, sequence_length, dim_model)

      tgt_mask = self.get_tgt_mask(src_act_size, tgt.shape[1]).to(tgt.device)
      memory_shape = [1, full_seqs.shape[1], full_seqs.shape[2]]
      memory = torch.zeros(memory_shape, device=full_seqs.device)
      # Transformer blocks - Out size = (batch_size, sequence_length, num_tokens)
      transformer_out = self.transformer(full_seqs, memory=memory, tgt_mask=tgt_mask)
      transformer_out = transformer_out.permute(1, 0, 2)

      next_token_logits = self.out(transformer_out[:, -1:])
      out_logits.append(next_token_logits)

      next_token = self.logits_to_state(next_token_logits)
      tgt = torch.cat([tgt, next_token.argmax(dim=-1)], dim=1)

    if return_logits:
      out_logits = torch.cat(out_logits, dim=1)
      return out_logits, torch.tensor(0), torch.tensor(1)

    return tgt[:, 1:], torch.tensor(0), torch.tensor(1)

  def prepare_acts(self, acts):
    if self.act_dtype == torch.long:
      acts = F.one_hot(acts, self.act_dim).float()
    return acts

  def forward(self, state, acts, tgt=None, tgt_mask=None,
              tgt_pad_mask=None, return_logits=False):
    # State size must be (batch_size, state sequence length)
    # Tgt size must be (batch_size, tgt sequence length)
    if tgt is None:
      return self._inference_forward(state, acts, return_logits=return_logits)

    # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
    adjusted_acts = acts + self.n_embeddings
    src = torch.cat([state, adjusted_acts.unsqueeze(1)], dim=1)
    src_act_size = src.shape[1]

    output_start_tokens = torch.full(
      (src.shape[0], 1), self.special_token, dtype=torch.long, device=src.device)

    full_seqs = torch.cat([src, output_start_tokens, tgt], dim=1)
    full_seqs = full_seqs[:, :-1]

    full_seqs = self.embedding(full_seqs) * math.sqrt(self.embedding_dim)
    full_seqs = self.positional_encoder(full_seqs)
    full_seqs = full_seqs.permute(1, 0, 2)
    # Size (batch_size, sequence_length, dim_model)

    # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
    memory_shape = [1, full_seqs.shape[1], full_seqs.shape[2]]
    memory = torch.zeros(memory_shape, device=full_seqs.device)
    transformer_out = self.transformer(
      full_seqs, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
    transformer_out = transformer_out.permute(1, 0, 2)
    tgt_length = tgt.shape[1]
    state_logits = self.out(transformer_out)[:, -tgt_length:]
    if return_logits:
      return state_logits, torch.tensor(0), torch.tensor(1)

    states = self.logits_to_state(state_logits)
    return states.argmax(dim=-1), torch.tensor(0), torch.tensor(1)
    
  def get_tgt_mask(self, src_act_size, tgt_size) -> torch.tensor:
    # Generates a square matrix where the each row allows one word more to be seen
    full_size = src_act_size + tgt_size
    # src_act_mask = torch.ones((src_act_size, tgt_size,))
    mask = torch.tril(torch.ones(full_size, full_size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    
    return mask