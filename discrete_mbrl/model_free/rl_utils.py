import argparse

import numpy as np

from data_logging import *


def make_mf_arg_parser():
  # These are the dyna-specific arguments only
  parser = argparse.ArgumentParser()
  parser.add_argument('--policy_hidden', nargs='*', default=[256, 256])
  parser.add_argument('--critic_hidden', nargs='*', default=[256, 256])
  parser.add_argument('--mf_steps', type=int, default=100_000)
  parser.add_argument('--epsilon', type=float, default=0.15)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--n_ae_updates', type=int, default=1)
  parser.add_argument('--ae_batch_size', type=int, default=None)
  parser.add_argument('--replay_size', type=int, default=100_000)
  parser.add_argument('--env_change_freq', default='-1') # 'episode'
  parser.add_argument('--env_change_type', default='random', choices=['random', 'next'])
  parser.add_argument('--rl_start_step', type=int, default=0)
  parser.add_argument('--rl_activation', default='relu', choices=['relu', 'crelu', 'tanh'])
  parser.add_argument('--ortho_init', action='store_true')

  # PPO arguments
  parser.add_argument('--ppo_batch_size', type=int, default=32)
  parser.add_argument('--ppo_iters', type=int, default=20)
  parser.add_argument('--ppo_clip', type=float, default=0.2)
  parser.add_argument('--ppo_value_coef', type=float, default=0.5)
  parser.add_argument('--ppo_entropy_coef', type=float, default=0.003)
  parser.add_argument('--ppo_gae_lambda', type=float, default=0) # 0.95 is recommended if using GAEs
  parser.add_argument('--ppo_norm_advantages', action='store_true')
  parser.add_argument('--ppo_max_grad_norm', type=float, default=0) # 0.5 recommended if used


  # Binary arguments
  parser.add_argument('--ae_recon_loss', action='store_true')
  parser.add_argument('--ae_recon_loss_binary', type=int, default=None)

  # Whether to sample from the ER to train the AE
  parser.add_argument('--ae_er_train', action='store_true')
  parser.add_argument('--ae_er_train_binary', type=int, default=None)

  parser.add_argument('--e2e_loss_binary', type=int, default=None)

  parser.set_defaults(ae_recon_loss=False, ppo_norm_advantages=False, ortho_init=False)
 
  return parser

def interpret_layer_sizes(sizes):
  if isinstance(sizes, (list, tuple)):
    if len(sizes) == 1:
      sizes = sizes[0]
    elif isinstance(sizes[0], int):
      return sizes
    elif isinstance(sizes[0], str):
      return [int(s) for s in sizes]

  if isinstance(sizes, (int, float)):
    return [int(sizes)]
  elif isinstance(sizes, str):
    # Check if single number or list of numbers
    return eval(sizes)
  else:
    raise ValueError(f'Invalid layer sizes format: {sizes}')

def epsilon_greedy_sample(model, obs, epsilon):
  """Samples an action from the model with epsilon-greedy exploration."""
  if np.random.rand() < epsilon:
    return np.random.randint(model.n_acts)
  else:
    return model.predict(obs)

def update_stats(stats, update_dict):
  for k, v in update_dict.items():
    stats[k].append(v)

def log_stats(stats, step, args):
  mean_stats = {k: np.mean(v) for k, v in stats.items()}

  # Create a pretty log string
  log_str = f'\n--- Step {step} ---\n'
  for i, (k, v) in enumerate(mean_stats.items()):
    log_str += f'{k}: {v:.3f}'
    if i < len(mean_stats) - 1:
      if i % 3 == 2:
        log_str += '\n'
      else:
        log_str += '  \t| '
  # print(log_str)

  # Remove nans for Wandb
  mean_stats = {k: v for k, v in mean_stats.items() if not np.isnan(v)}
  mean_stats['step'] = step
  log_metrics(mean_stats, args, step=step)
  
def to_device(tensors, device):
  return [t.to(device) for t in tensors]