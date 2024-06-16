import gc
import psutil
import os
import sys
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from shared.models import *
from shared.trainers import *
from data_helpers import *
from data_logging import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from utils import *
from eval_policies.policies import load_policy

sns.set()

GAMMA_CONST = 0.99
N_EXAMPLE_IMGS = 15
SEED = 0 # Should be same as seed used for prior steps
PRELOAD_TEST = False
TEST_WORKERS = 0
EARLY_STOP_COUNT = 3000
DISCRETE_TRANS_TYPES = ('discrete', 'transformer', 'transformerdec')
CONTINUOUS_TRANS_TYPES = ('continuous', 'shared_vq')
N_RAND_LATENT_SAMPLES = 500
STATE_DISTRIB_SAMPLES = 20000 # 20000
IMAGINE_DISTRIB_SAMPLES = 2000 # 2000
UNIQUE_OBS_EARLY_STOP = 1.0 # 0.2s


def calculate_trans_losses(
  next_z, next_reward, next_gamma, next_z_pred_logits, next_z_pred, next_reward_pred,
  next_gamma_pred, next_obs, trans_model_type, encoder_model, rand_obs=None,
  init_obs=None, all_obs=None, all_trans=None, curr_z=None, acts=None):
  # Calculate the transition reconstruction loss
  loss_dict = {}
  if trans_model_type in CONTINUOUS_TRANS_TYPES:
    assert next_z.shape == next_z_pred_logits.shape
    state_losses = torch.pow(next_z - next_z_pred_logits, 2)
    state_losses = state_losses.view(next_z.shape[0], -1).sum(1)
    loss_dict['state_loss'] = state_losses.cpu().numpy()
    loss_dict['state_acc'] = np.array([0] * next_z.shape[0])
  elif trans_model_type in DISCRETE_TRANS_TYPES:
    state_losses = F.cross_entropy(
      next_z_pred_logits, next_z, reduction='none')
    state_losses = state_losses.view(next_z.shape[0], -1).sum(1)
    state_accs = (next_z_pred == next_z).float().view(next_z.shape[0], -1).mean(1)
    loss_dict['state_loss'] = state_losses.cpu().numpy()
    loss_dict['state_acc'] = state_accs.cpu().numpy()

  # Calculate the transition image reconstruction loss
  next_obs_pred = encoder_model.decode(next_z_pred).cpu()
  img_mse_losses = torch.pow(next_obs - next_obs_pred, 2)
  loss_dict['img_mse_loss'] = img_mse_losses.view(next_obs.shape[0], -1).sum(1).numpy()
  loss_dict['reward_loss'] = F.mse_loss(next_reward,
    next_reward_pred.squeeze().cpu(), reduction='none').numpy()
  loss_dict['gamma_loss'] = F.mse_loss(next_gamma,
    next_gamma_pred.squeeze().cpu(),reduction='none').numpy()

  if rand_obs is not None:
    rand_img_mse_losses = torch.pow(next_obs - rand_obs, 2)
    loss_dict['rand_img_mse_loss'] = rand_img_mse_losses.view(
      next_obs.shape[0], -1).sum(1).numpy()
  else:
    loss_dict['rand_img_mse_loss'] = np.array([np.nan] * next_obs.shape[0])
    
  if init_obs is not None:
    init_img_mse_losses = torch.pow(next_obs - init_obs, 2)
    loss_dict['init_img_mse_loss'] = init_img_mse_losses.view(
      next_obs.shape[0], -1).sum(1).numpy()
  else:
    loss_dict['init_img_mse_loss'] = np.array([np.nan] * next_obs.shape[0])

  if all_obs is not None:
    no_dists, no_idxs = get_min_mses(next_obs_pred, all_obs, return_idxs=True)
    loss_dict['closest_img_mse_loss'] = no_dists

    if all_trans is not None and curr_z is not None:
      curr_obs_pred = encoder_model.decode(curr_z).cpu()
      o_dists, o_idxs = get_min_mses(curr_obs_pred, all_obs, return_idxs=True)
      start_obs = to_hashable_tensor_list(all_obs[o_idxs])
      end_obs = to_hashable_tensor_list(all_obs[no_idxs])
      acts = acts.cpu().tolist()

      trans_exists = []
      for so, eo, a in zip(start_obs, end_obs, acts):
        if all_trans[so][(a, eo)] == 0:
          trans_exists.append(0)
        else:
          trans_exists.append(1)

      loss_dict['real_transition_frac'] = np.array(trans_exists)
    else:
      loss_dict['real_transition_frac'] = np.array([np.nan] * next_obs.shape[0])
  else:
    loss_dict['closest_img_mse_loss'] = np.array([np.nan] * next_obs.shape[0])
    loss_dict['real_transition_frac'] = np.array([np.nan] * next_obs.shape[0])

  return loss_dict

def wandb_log(items, do_log, make_gif=False):
  if do_log:
    for k, v in items.items():
      if isinstance(v, list) and isinstance(v[0], np.ndarray) \
        and len(v[0].shape) > 1:
        items[k] = [wandb.Image(x) for x in v]
      elif isinstance(v, np.ndarray) and len(v.shape) > 1:
        if make_gif:
          items[k] = wandb.Video(v, fps=4, format='gif')
        else:
          items[k] = wandb.Image(v)
      elif isinstance(v, Figure):
        items[k] = wandb.Image(v)
    wandb.log(items)

def save_and_log_imgs(imgs, label, results_dir, args):
  # if args.save:
  #   plt.savefig(os.path.join(results_dir,
  #     f'{args.ae_model_type}_v{args.ae_model_version}_{label}.png'))
  wandb_log({label: [img.clip(0, 1) for img in imgs]}, args.wandb)

def get_min_mses(gens, sources, return_idxs=False):
  """ Get the minimum distance between each generated sample and all sources. """
  if len(gens.shape) == len(sources.shape) - 1:
    gens = gens[None]
  assert gens.shape[1:] == sources.shape[1:], \
    f'gens.shape: {gens.shape}, sources.shape: {sources.shape}, but core dims need to be equal!'

  min_dists = []
  min_idxs = []
  for gen in gens:
    dists = (gen.unsqueeze(0) - sources) ** 2
    dists = dists.reshape(dists.shape[0], -1).sum(dim=1)
    min_dist = dists.min().item()
    min_dists.append(min_dist)
    if return_idxs:
      min_idx = dists.argmin().item()
      min_idxs.append(min_idx)
  if return_idxs:
    return np.array(min_dists), np.array(min_idxs)
  return np.array(min_dists)

def update_losses(losses, new_losses, args, step, log=True):
  for k, v in new_losses.items():
    losses[k].extend(v)
  n_losses = len(new_losses[k])
  losses['model'].extend([f'{args.trans_model_type}' \
    + f'_v{args.trans_model_version}' for _ in range(n_losses)])
  losses['step'].extend([step for _ in range(n_losses)])

def eval_model(args, encoder_model=None, trans_model=None):
  import_logger(args)

  # Randomize pytorch seed
  # Doing this because I was getting unintentially deterministic behavior
  torch.manual_seed(time.time())

  # Collect a set of all unique observations in the environment
  # Not recommended for complex environments
  if args.exact_comp:
    unique_obs, unique_data_hash = get_unique_obs(
      args, cache=True, partition='all', return_hash=True,
      early_stop_frac=UNIQUE_OBS_EARLY_STOP)

    log_metrics({'unique_obs_count': len(unique_obs)}, args)
  else:
    unique_obs = None
  trans_dict = None # Currently not used

  env = make_env(args.env_name, max_steps=args.env_max_steps)
  

  ### Loading Models & Some Data ###


  print('Loading data...')
  test_loader = prepare_dataloader(
    args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
    randomize=False, n=args.max_transitions, n_preload=TEST_WORKERS, preload=args.preload_data,
    extra_buffer_keys=args.extra_buffer_keys)
  test_sampler = create_fast_loader(
    test_loader.dataset, batch_size=1, shuffle=True, num_workers=TEST_WORKERS, n_step=1)
  rev_transform = test_loader.dataset.flat_rev_obs_transform

  # Load the encoder
  if encoder_model is None:
    sample_obs = next(iter(test_sampler))[0]
    encoder_model = construct_ae_model(
      sample_obs.shape[1:], args)[0]
  encoder_model = encoder_model.to(args.device)
  freeze_model(encoder_model)
  encoder_model.eval()
  print(f'Loaded encoder')
    
  if hasattr(encoder_model, 'enable_sparsity'):
    encoder_model.enable_sparsity()

  # Load the transition model
  if trans_model is None:
    trans_model = construct_trans_model(encoder_model, args, env.action_space)[0]
  trans_model = trans_model.to(args.device)
  freeze_model(trans_model)
  trans_model.eval()
  print(f'Loaded transition model')


  # Hack for universal_vq to work with the current code
  if args.trans_model_type == 'universal_vq':
    if encoder_model.quantized_enc:
      global CONTINUOUS_TRANS_TYPES
      CONTINUOUS_TRANS_TYPES = CONTINUOUS_TRANS_TYPES + ('universal_vq',)
    else:
      global DISCRETE_TRANS_TYPES
      DISCRETE_TRANS_TYPES = DISCRETE_TRANS_TYPES + ('universal_vq',)


  ### Exact State Comparison ###


  if args.exact_comp:
    
    ### Log the representations of each state ###
    
    if args.log_state_reprs:
      print('Logging state representations...')
      # First sort unique obs based on hashes, then encode in batches
      hashes = hash_tensors(unique_obs)
      order = np.argsort(hashes)
      ordered_obs = unique_obs[order]
      
      state_reprs = []
      for i in range(0, len(ordered_obs), args.eval_batch_size):
        reprs = encoder_model.encode(
          ordered_obs[i:i+args.eval_batch_size].to(args.device))
        if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
          reprs = reprs.reshape(reprs.shape[0], encoder_model.latent_dim)
        state_reprs.extend(list(reprs.cpu().detach().numpy()))
      state_reprs = np.stack(state_reprs)
      
      log_np_array(state_reprs, 'state_reprs', args)

    ### Start getting state distributions ###

    vec_env = DummyVecEnv([lambda: FreezeOnDoneWrapper(make_env(args.env_name, max_steps=args.env_max_steps)) \
      for _ in range(args.eval_batch_size)])

    for eval_policy in args.eval_policies:
      target_state_ids = [[] for _ in range(args.eval_unroll_steps)]
      pred_state_ids = [[] for _ in range(args.eval_unroll_steps)]
      n_unknown_states = 0
      total_iter_states = 0

      # First check for cached version of stated distribution
      state_distr_hash = unique_data_hash + '_' + eval_policy + '_' + \
        str(args.eval_unroll_steps) + '_' + str(STATE_DISTRIB_SAMPLES)
      state_distr_path = os.path.join(CACHE_DIR, f'{state_distr_hash}.pkl')
      if os.path.exists(state_distr_path):
        print(f'Loading cached state distribution data from {state_distr_path}...')
        time_since_update = time.time() - os.path.getmtime(state_distr_path)
        while time_since_update < 30:
          time.sleep(10)
          time_since_update = time.time() - os.path.getmtime(state_distr_path)
        with open(state_distr_path, 'rb') as f:
          state_distribs = pickle.load(f)
        print('Finished loading!')
      else:
        print(f'No cached state distribution data found, generating new data')
        state_distribs = None

      print('Simulating trajectories to calculate state visitation frequencies...')
      
      policy = load_policy(args.env_name, eval_policy)
      print(f'Loaded {eval_policy} policy', flush=True)
      policy = policy.to(args.device)
      print(f'Pushing policy to device', flush=True)
      policy.eval()
      print(f'Put policy in eval mode', flush=True)


      ### First handle real trajectories ###


      n_total_states = len(unique_obs)

      print('Simulating runs for real state distribution...')
      if state_distribs is None:
        # Loop over a batch of trajectories
        n_batches = int(np.ceil(STATE_DISTRIB_SAMPLES / args.eval_batch_size))
        for batch_idx in tqdm(range(n_batches)):
          transitions = [[], [], []]
          obs = vec_env.reset()
          for _ in range(args.eval_unroll_steps):
            fobs = torch.from_numpy(obs).float().to(args.device)
            act = policy.act(fobs).cpu().tolist()
            next_obs, _, done, infos = vec_env.step(act)

            for i, e in enumerate([obs, act, next_obs]):
              transitions[i].append(e)
            obs = next_obs

          # Format trajectory data as tensors
          transitions[0] = torch.from_numpy(np.stack(transitions[0])).to(torch.float16).float()
          transitions[0] = transitions[0].transpose(0, 1)
          transitions[1] = torch.from_numpy(np.stack(transitions[1])).long()
          transitions[2] = torch.from_numpy(np.stack(transitions[2])).to(torch.float16).float()
          transitions[2] = transitions[2].transpose(0, 1)
        
          # Convert observations into state IDs
          target_obs = transitions[2]
          batch_target_dists, batch_target_idxs = get_min_mses(
            target_obs.reshape(-1, *target_obs.shape[2:]), unique_obs, return_idxs=True)
          batch_target_dists = batch_target_dists.reshape(target_obs.shape[:2])
          batch_target_idxs = batch_target_idxs.reshape(target_obs.shape[:2])

          for target_dists, target_idxs in zip(batch_target_dists, batch_target_idxs):
            for i, (dist, idx) in enumerate(zip(target_dists, target_idxs)):
              if dist < EPSILON:
                target_state_ids[i].append(idx)
              else:
                n_unknown_states += 1
            total_iter_states += len(target_idxs)
            
        # Calculate real state visitation frequencies if not cached
        state_distribs = []
        for i in range(args.eval_unroll_steps):
          target_ids = torch.tensor(target_state_ids[i]).long()
          target_distrib = F.one_hot(target_ids, n_total_states).float().mean(dim=0)
          state_distribs.append(target_distrib)
        with open(state_distr_path, 'wb') as f:
          pickle.dump(state_distribs, f)
        print(f'Saved state distribution data to {state_distr_path}')


      ### Next handle imagined trajectories ###


      print('Simulating runs for imagined state distribution...')
      n_batches = int(np.ceil(IMAGINE_DISTRIB_SAMPLES / args.eval_batch_size))
      for batch_idx in tqdm(range(n_batches)):
        pred_states = []
        obs = vec_env.reset()
        curr_states = encoder_model.encode(torch.from_numpy(obs) \
          .float().to(args.device))

        # Predict all states from the starting state
        frozen_idxs = -torch.ones((len(obs),), device=args.device)
        for i in range(args.eval_unroll_steps):
          pred_obs = encoder_model.decode(curr_states)
          act = policy.act(pred_obs)
          curr_states, _, gamma = trans_model(curr_states, act)

          # TODO: Add gamma prediction to transformers and fix terminal predictions
          # Freeze states that are predicted to be terminal
          for j in range(len(frozen_idxs)):
            if frozen_idxs[j] != -1 and i > 0:
              curr_states[j] = pred_states[-1][j]
              
          if gamma is not None:
            done_envs = gamma.squeeze(1) < 0.5
            not_done_idxs = frozen_idxs == -1
            update_idxs = torch.logical_and(done_envs, not_done_idxs)
            frozen_idxs[update_idxs] = i

          pred_states.append(curr_states)
        
        # (seq x batch) x obs
        pred_states = torch.cat(pred_states)
        pred_obs = encoder_model.decode(pred_states).cpu()
        # (seq x batch)
        pred_idxs = get_min_mses(pred_obs, unique_obs, return_idxs=True)[1]
        pred_idxs = rearrange(
          pred_idxs, '(s b) -> s b',
          b=args.eval_batch_size, s=args.eval_unroll_steps)
        
        for i in range(len(pred_idxs)):
          pred_state_ids[i].extend(pred_idxs[i])
      
      if total_iter_states > 0:
        frac_known_states = 1 - n_unknown_states / total_iter_states
        print(f'Fraction of known states: {frac_known_states:.3f}')
        log_metrics({f'{eval_policy}_frac_known_states': frac_known_states}, args)

      uniform_distrib = torch.ones(n_total_states) / n_total_states

      # Calculate sample KL-divergence at each step and log
      print('Calculating state distributions...')
      kl_divs = []
      pred_distribs = []
      for i in range(args.eval_unroll_steps):
        pred_ids = torch.tensor(pred_state_ids[i]).long()
        pred_distrib = F.one_hot(pred_ids, n_total_states).float().mean(dim=0)
        pred_distribs.append(pred_distrib)
        
        kl_div = (state_distribs[i] * (state_distribs[i] / \
          (pred_distrib + EPSILON) + EPSILON).log()).sum().item()
        kl_divs.append(kl_div)
        first_state_kl_div = (state_distribs[i] * (state_distribs[i] / \
          (state_distribs[0] + EPSILON) + EPSILON).log()).sum().item()
        delayed_state_kl_div = (state_distribs[i] * (state_distribs[i] / \
          (state_distribs[max(0, i-1)] + EPSILON) + EPSILON).log()).sum().item()
        uniform_kl_div = (state_distribs[i] * (state_distribs[i] / \
          uniform_distrib + EPSILON).log()).sum().item()

        print('KL-divergence at step {}: {:.3f}'.format(i + 1, kl_div))
        log_metrics({
          f'{eval_policy}_state_distrib_kl_div': kl_div,
          f'{eval_policy}_first_state_distrib_kl_div': first_state_kl_div,
          f'{eval_policy}_delayed_state_distrib_kl_div': delayed_state_kl_div,
          f'{eval_policy}_uniform_distrib_kl_div': uniform_kl_div,
          'n_step': i + 1}, args, step=i+1)
      log_metrics({f'{eval_policy}_state_distrib_kl_div_mean': np.mean(kl_divs)}, args)

      del target_state_ids, pred_state_ids


      ### Visualize the state distributions ###

      # First calculate the background image

      # I think this simplistic method only works because of the fact that
      # we are working with a grid, may need to change in the future
      background_obs = unique_obs.mode(dim=0).values
      img_diffs = unique_obs - background_obs

      distrib_imgs = []
      for i, (state_distrib, pred_distrib) in enumerate(zip(state_distribs, pred_distribs)):
        state_distrib_diffs = (state_distrib[:, None, None, None] \
          * img_diffs).sum(dim=0).numpy()
        pred_distrib_diffs = (pred_distrib[:, None, None, None] \
          * img_diffs).sum(dim=0).numpy()
        # Normalize so it's easier to see
        norm_factor = 1 / state_distrib_diffs.max()
        norm_state_diffs = state_distrib_diffs * norm_factor
        state_distrib_obs = background_obs + norm_state_diffs
        state_distib_img = state_distrib_obs.permute(1, 2, 0).numpy()

        norm_factor = 1 / pred_distrib_diffs.max()
        norm_pred_diffs = pred_distrib_diffs * norm_factor
        pred_distrib_obs = background_obs + norm_pred_diffs
        pred_distib_img = pred_distrib_obs.permute(1, 2, 0).numpy()

        norm_error_diffs = norm_state_diffs - norm_pred_diffs
        rev_norm_error_diffs = norm_pred_diffs - norm_state_diffs
        red_channel = np.copy(rev_norm_error_diffs[0])
        rev_norm_error_diffs[0] = rev_norm_error_diffs[1]
        rev_norm_error_diffs[1] = rev_norm_error_diffs[2]
        rev_norm_error_diffs[2] = red_channel
        error_distrib_obs = background_obs + norm_error_diffs + rev_norm_error_diffs
        error_distrib_img = error_distrib_obs.permute(1, 2, 0).numpy()

        # Ground truth, prediction, difference
        distrib_img = np.concatenate(
          [state_distib_img, pred_distib_img, error_distrib_img], axis=0)
        distrib_imgs.append(distrib_img)
      temporal_distrib_img = np.concatenate(distrib_imgs, axis=1)
      temporal_distrib_img = temporal_distrib_img.clip(0, 1)
      log_images({f'{eval_policy}_temporal_state_distrib': [temporal_distrib_img]}, args)
      

      ### Visualize the state distributions with plots ###


      # Order from most to least frequent in final ground truth distribution
      state_order = state_distribs[-1].numpy().argsort()[::-1]

      fig = plt.figure(figsize=(len(unique_obs) / 168 * 14, args.eval_unroll_steps))
      axs = fig.subplots(args.eval_unroll_steps, 1)
      plots = []
      for i in range(args.eval_unroll_steps):
        r1 = state_distribs[i].numpy()[state_order]
        r2 = pred_distribs[i].numpy()[state_order]
        
        p1 = sns.barplot(
          x=list(range(len(r1))), y=r1, color='red', alpha=0.5, ax=axs[i])
        p2 = sns.barplot(
          x=list(range(len(r2))), y=r2, color='blue', alpha=0.5, ax=axs[i])
        plots.extend([p1, p2])

        axs[i].set_ylabel(f'{i+1}')

        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i].set_xlim([0, len(unique_obs)])

      fig.text(0.5, 0.08, 'State Visitation Distribution', ha='center')
      fig.text(0.08, 0.5, 'Step', va='center', rotation='vertical')
      fig.suptitle('State Visitation Distributions', fontsize=16)
      fig.tight_layout(rect=[0.1, 0.09, 1, 0.98])

      legend = plots[0].legend(
        [plots[0].patches[0], plots[1].patches[0]],
        ['Ground Truth', 'Simulated'],
        loc='upper right',
        bbox_to_anchor=(0.9, 1),
        ncol=2,
        frameon=True)

      legend.legendHandles[0].set_color('red')
      legend.legendHandles[1].set_color('blue')

      log_figures({f'{eval_policy}_state_distrib_plot': [fig]}, args)

    
  ### End Exact State Comparison ###


  results_dir = f'./results/{args.env_name}'
  os.makedirs(results_dir, exist_ok=True)

  torch.manual_seed(SEED)
  gc.collect()
  print('Memory usage:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)


  ### Encoder Testing ###


  # Calculate autoencoder reconstruction loss
  n_samples = 0
  encoder_recon_loss = torch.tensor(0, dtype=torch.float64)
  all_latents = []
  for batch_data in test_loader:
    obs_data = batch_data[0]
    n_samples += obs_data.shape[0]
    with torch.no_grad():
      latents = encoder_model.encode(obs_data.to(args.device))
      recon_outputs = encoder_model.decode(latents)
      all_latents.append(latents.cpu())
    encoder_recon_loss += torch.sum((recon_outputs.cpu() - obs_data)**2)
  all_latents = torch.cat(all_latents, dim=0)
  encoder_recon_loss = (encoder_recon_loss / n_samples).item()
  print(f'Encoder reconstruction loss: {encoder_recon_loss:.2f}')
  log_metrics({'encoder_recon_loss': encoder_recon_loss}, args)

  gc.collect()
  print('Memory usage:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

  # # Calculate symmetic uncertainty of discrete latents
  # if args.trans_model_type in DISCRETE_TRANS_TYPES:
  #   # This takes a large amount of memory
  #   sym_uncertainty = sample_symmetric_uncertainty(all_latents)
  #   import psutil; print('a:', psutil.virtual_memory()[3]/1e9)
  #   mean_su = triu_avg(sym_uncertainty) # Avg SU of each latent variable pair
  #   import psutil; print('a:', psutil.virtual_memory()[3]/1e9)
  #   print(f'Mean symmetrical uncertainty: {mean_su:.3f}')
  #   log_metrics({'mean_symmetrical_uncertainty': mean_su}, args)
  #   # # Create heatmap of SU for visualization
  #   # if args.wandb:
  #   #   x_labels = y_labels = list(range(sym_uncertainty.shape[0]))
  #   #   wandb.log({'symmetrical_uncertainty': wandb.plots.HeatMap(
  #   #     x_labels, y_labels, sym_uncertainty.cpu().numpy(), show_text=False
  #   #   )})


  # Sample random latent vectors eval
  print('Sampling random latent vectors...')
  if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
    latent_dim = encoder_model.latent_dim
    all_latents = all_latents.reshape(all_latents.shape[0], latent_dim)

    latent_min = all_latents.min()
    latent_max = all_latents.max()
    latent_range = latent_max - latent_min
    uniform_sampled_latents = torch.rand((N_RAND_LATENT_SAMPLES, latent_dim))
    uniform_sampled_latents = uniform_sampled_latents * latent_range + latent_min

    with torch.no_grad():
      obs = encoder_model.decode(uniform_sampled_latents.to(args.device))
    obs = obs.cpu()
    
    if args.exact_comp:
      min_dists = get_min_mses(obs, unique_obs)
      print('uniform_min_l2:', min_dists.mean())
      log_metrics({'uniform_cont_sample_latent_obs_l2': min_dists.mean()}, args)
    imgs = obs_to_img(obs[:N_EXAMPLE_IMGS], env_name=args.env_name, rev_transform=rev_transform)
    log_images({'uniform_cont_sample_latent_imgs': imgs}, args)

    latent_means = all_latents.mean(dim=0)
    print('latent_means (16):', latent_means.shape)
    latent_stds = all_latents.std(dim=0)
    normal_sampled_latents = torch.normal(
      latent_means.repeat(N_RAND_LATENT_SAMPLES),
      latent_stds.repeat(N_RAND_LATENT_SAMPLES))
    normal_sampled_latents = normal_sampled_latents.reshape(N_RAND_LATENT_SAMPLES, latent_dim)
    print(normal_sampled_latents.shape)
    with torch.no_grad():
      obs = encoder_model.decode(normal_sampled_latents.to(args.device))
    obs = obs.cpu()
    if args.exact_comp:
      min_dists = get_min_mses(obs, unique_obs)
      print('normal_min_l2:', min_dists.mean())
      log_metrics({'normal_sample_latent_obs_l2': min_dists.mean()}, args)
    imgs = obs_to_img(obs[:N_EXAMPLE_IMGS], env_name=args.env_name, rev_transform=rev_transform)
    log_images({'normal_sample_latent_imgs': imgs}, args)

  elif args.trans_model_type in DISCRETE_TRANS_TYPES:
    latent_dim = encoder_model.n_latent_embeds
    sampled_latents = torch.randint(
      0, encoder_model.n_embeddings, (N_RAND_LATENT_SAMPLES, latent_dim,))
    with torch.no_grad():
      obs = encoder_model.decode(sampled_latents.to(args.device))
    obs = obs.cpu()
    if args.exact_comp:
      min_dists = get_min_mses(obs, unique_obs)
      print('uniform_min_l2:', min_dists.mean())
      log_metrics({'uniform_disc_sample_latent_obs_l2': min_dists.mean()}, args)
    imgs = obs_to_img(obs[:N_EXAMPLE_IMGS], env_name=args.env_name, rev_transform=rev_transform)
    log_images({'uniform_disc_sample_latent_imgs': imgs}, args)


  # Generate reconstruction sample images
  print('Generating reconstruction sample images...')
  example_imgs = []
  for i, sample_transition in enumerate(test_sampler):
    sample_obs = sample_transition[0]
    if i >= N_EXAMPLE_IMGS:
      break
    with torch.no_grad():
      recon_obs = encoder_model(sample_obs.to(args.device))
    if isinstance(recon_obs, tuple):
      recon_obs = recon_obs[0]
    both_obs = torch.cat([sample_obs, recon_obs.cpu()], dim=0)
    both_imgs = obs_to_img(both_obs, env_name=args.env_name, rev_transform=rev_transform)
    cat_img = np.concatenate([both_imgs[0], both_imgs[1]], axis=1)
    example_imgs.append(cat_img)
    
  recon_img_arr = np.concatenate(example_imgs, axis=1)
  plt.figure(figsize=(N_EXAMPLE_IMGS*2, N_EXAMPLE_IMGS))
  plt.imshow(recon_img_arr.clip(0, 1))
  log_images({'recon_sample_imgs': example_imgs}, args)

  del test_loader, test_sampler

  gc.collect()
  print('Memory usage:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)


  ### Transition Model Testing ###

  # test_workers = 0 if PRELOAD_TEST else int(args.n_preload/3)
  n_step_loader = prepare_dataloader(
    args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
    randomize=True, n=args.max_transitions, n_preload=TEST_WORKERS, preload=args.preload_data,
    n_step=args.eval_unroll_steps, extra_buffer_keys=args.extra_buffer_keys)
  # alt_n_step_loader = create_fast_loader(
  #   n_step_loader.dataset, batch_size=args.batch_size,
  #   shuffle=True, num_workers=TEST_WORKERS, n_step=args.eval_unroll_steps)
  n_step_sampler = create_fast_loader(
    n_step_loader.dataset, batch_size=1, shuffle=True, num_workers=TEST_WORKERS,
    n_step=args.eval_unroll_steps)
    
  print(f'Sampled {args.eval_unroll_steps}-step sub-trajectories')


  # Caculate n-step statistics
  n_step_stats = dict(
    state_loss=[], state_acc=[], reward_loss=[],
    gamma_loss=[], img_mse_loss=[], rand_img_mse_loss=[],
    init_img_mse_loss=[], step=[], model=[],
    closest_img_mse_loss=[], real_transition_frac=[])

  n_full_unroll_samples = 0
  print('Calculating stats for n-step data...')
  for i, n_step_trans in tqdm(enumerate(n_step_loader), total=len(n_step_loader)):
    obs, acts, next_obs, rewards, dones = n_step_trans[:5]
    # rand_obs = next(iter(alt_n_step_loader))[0][:obs.shape[0]]
    # while rand_obs.shape[0] < obs.shape[0]:
    #   rand_obs = next(iter(alt_n_step_loader))[0][:obs.shape[0]]
    rand_obs = None
    gammas = (1 - dones) * GAMMA_CONST
    z = encoder_model.encode(obs[:, 0].to(args.device))
    if args.trans_model_type in DISCRETE_TRANS_TYPES:
      z_logits = F.one_hot(z, encoder_model.n_embeddings).permute(0, 2, 1).float() * 1e6
    else:
      z = z.reshape(z.shape[0], encoder_model.latent_dim)
      z_logits = z

    loss_dict = \
      calculate_trans_losses(z, rewards[:, 0], gammas[:, 0], z_logits, z,
      rewards[:, 0], gammas[:, 0], obs[:, 0], args.trans_model_type, encoder_model,
      rand_obs=rand_obs[:, 0] if rand_obs else None, init_obs=obs[:, 0], all_obs=unique_obs)
    update_losses(n_step_stats, loss_dict, args, 0)

    keep_idxs = set(range(obs.shape[0]))
    for step in range(args.eval_unroll_steps):
      next_z = encoder_model.encode(next_obs[:, step].to(args.device))
      if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
        next_z = next_z.reshape(next_z.shape[0], encoder_model.latent_dim)

      next_z_pred_logits, next_reward_pred, next_gamma_pred = \
        trans_model(z, acts[:, step].to(args.device), return_logits=True)
      next_z_pred = trans_model.logits_to_state(next_z_pred_logits)
      if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
        next_z_pred_logits = next_z_pred_logits.reshape(
          next_z_pred_logits.shape[0], encoder_model.latent_dim)

      loss_dict = calculate_trans_losses(
          next_z, rewards[:, step], gammas[:, step],
          next_z_pred_logits, next_z_pred, next_reward_pred, next_gamma_pred,
          next_obs[:, step], args.trans_model_type, encoder_model,
          rand_obs=rand_obs[:next_obs.shape[0], step] if rand_obs else None,
          init_obs=obs[:, 0], all_obs=unique_obs, all_trans=trans_dict,
          curr_z=z, acts=acts[:, step])
      update_losses(n_step_stats, loss_dict, args, step+1)

      z = next_z_pred

      # Remove transitions with finished episodes
      keep_idxs = (dones[:, step] == 0).float().nonzero().squeeze()
      if keep_idxs.numel() == 0:
        break
      obs, acts, next_obs, rewards, dones = \
        [x[keep_idxs] for x in (obs, acts, next_obs, rewards, dones)]
      gammas = gammas[keep_idxs]
      z = z[keep_idxs]

    n_full_unroll_samples += keep_idxs.numel()
    print(f'{n_full_unroll_samples}/{EARLY_STOP_COUNT} full trajectories sampled')
    if n_full_unroll_samples >= EARLY_STOP_COUNT:
      break


  # Upload the stats to logging server
  print('Publishing n-step stats to cloud...')
  for step in range(args.eval_unroll_steps + 1):
    keep_idxs = [i for i, n_step in enumerate(n_step_stats['step']) \
      if n_step == step]
    log_vars = {k: np.nanmean(np.array(v)[keep_idxs]) \
      for k, v in n_step_stats.items() \
      if k not in ('step', 'model')}
    log_metrics({
      'n_step': step,
      **log_vars
    }, args, step=step)

  log_metrics({'img_mse_loss_mean': np.mean(n_step_stats['img_mse_loss'])}, args)


  gc.collect()
  print('Memory usage:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)


  # Create sample transition images
  print('Creating sample transition images...')
  samples = []
  for i, sample_rollout in enumerate(n_step_sampler):
    if len(samples) >= N_EXAMPLE_IMGS:
      break
    if sample_rollout[0].numel() > 0:
      samples.append(sample_rollout)
  sample_rollouts = [torch.stack([x[i] for x in samples]).squeeze(dim=1) \
    for i in range(len(samples[0]))]

  all_obs = torch.cat((sample_rollouts[0][:,:1], sample_rollouts[2]), dim=1)
  acts = sample_rollouts[1]
  dones = sample_rollouts[4]
  z = encoder_model.encode(all_obs[:, 0].to(args.device))
  if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
    z = z.reshape(z.shape[0], encoder_model.latent_dim)
  
  # Convert hidden states to observations (if necessary)
  all_obs = [states_to_imgs(o, args.env_name, transform=rev_transform) for o in all_obs]
  all_obs = torch.from_numpy(np.stack(all_obs))

  example_trans_imgs = []
  example_trans_imgs.append(torch.cat((
    all_obs[:, 0], torch.zeros_like(all_obs[:, 0])), dim=3))
  
  continue_mask = torch.ones(all_obs.shape[0])
  for step in range(args.eval_unroll_steps):
    z = trans_model(z, acts[:, step].to(args.device))[0]
    pred_obs = encoder_model.decode(z).cpu()
    
    pred_obs = states_to_imgs(pred_obs, args.env_name, transform=rev_transform)
    pred_obs = torch.from_numpy(pred_obs)
    
    example_trans_imgs.append(torch.cat((
      all_obs[:, step+1], pred_obs), dim=3) \
      * continue_mask[:, None, None, None])
    continue_mask[dones[:, step].float().nonzero().squeeze()] = 0
  example_trans_imgs = [
    torch.stack([x[i] for x in example_trans_imgs])
    for i in range(len(example_trans_imgs[0]))
  ]
  
  for i, img in enumerate(example_trans_imgs):
    img = (img.clip(0, 1) * 255).numpy().astype(np.uint8)
    grayscale = img.shape[1] == 2 or img.shape[1] > 3
    if grayscale:
      img = img[:, :-1, :, :]
    
    log_videos({f'{args.eval_unroll_steps}-step_transition_sample': [img]}, args)
    
    # img = img.permute(1, 2, 0).clip(0, 1).numpy()
    # grayscale = img.shape[2] == 2 or img.shape[2] > 3
    # if grayscale:
    #   img = img[:, :, -1]
      
    # plt.imshow(img)
    # if args.save:
    #   plt.savefig(os.path.join(results_dir,
    #     f'{args.trans_model_type}_trans_model_v{args.trans_model_version}' + \
    #     f'_{args.eval_unroll_steps}-step_sample_{i}.png'))
    
    # wandb_log({f'{args.eval_unroll_steps}-step_transition_sample': img}, args.wandb)

if __name__ == '__main__':
  # Parse args
  args = get_args()
  # Setup wandb
  args = init_experiment('discrete-mbrl-eval', args)
  # Evaluate the models
  eval_model(args)
  # Clean up wandb
  finish_experiment(args)
