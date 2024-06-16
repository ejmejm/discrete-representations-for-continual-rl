import argparse
from argparse import Namespace
import os
import psutil

from einops import rearrange
from gym.envs.mujoco import MujocoEnv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from visualization import states_to_imgs
# from model_construction import CONTINUOUS_ENCODER_TYPES, DISCRETE_ENCODER_TYPES, add_model_args
from env_helpers import check_env_name


def make_argparser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--env_name', type=str, default='MiniGrid-MultiRoom-N2-S4-v0')
    parser.add_argument('-t', '--ae_model_type', type=str, default='ae')
    parser.add_argument('-v', '--ae_model_version', type=str, default='2')
    parser.add_argument('-tet', '--trans_model_type', type=str, default='continuous')
    parser.add_argument('-tev', '--trans_model_version', type=str, default='1')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-ne', '--epochs', type=int, default=100)
    parser.add_argument('-tne', '--trans_epochs', type=int, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-l', '--log_freq', type=int, default=100)
    parser.add_argument('-c', '--checkpoint_freq', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
    parser.add_argument('-tlr', '--trans_learning_rate', type=float, default=None)
    parser.add_argument('-m', '--max_transitions', type=int, default=None)
    parser.add_argument('-rlc', '--recon_loss_clip', type=float, default=0)
    parser.add_argument('-u', '--all_data', dest='unique_data', action='store_false')
    parser.add_argument('--unique_data', dest='unique_data', action='store_true')
    parser.add_argument('-nl', '--no_load', dest='load', action='store_false')
    parser.add_argument('-nc', '--no_cache', dest='cache', action='store_false')
    parser.add_argument('-p', '--preprocess', action='store_true')
    parser.add_argument('-w', '--wandb', action='store_true')
    parser.add_argument('-cml', '--comet_ml', action='store_true')
    parser.add_argument('-i', '--extra_info', type=str, default=None)
    parser.add_argument('-pl', '--n_preload', type=int, default=0, help='Parallel preload data')
    parser.add_argument('-um', '--upload_model', action='store_true')
    parser.add_argument('-s', '--save', dest='save', action='store_true')
    parser.add_argument('-ntu', '--n_train_unroll', type=int, default=4)
    parser.add_argument('-pld', '--preload_data', action='store_true')
    parser.add_argument('--env_max_steps', type=int, default=None)
    parser.add_argument('--rl_unroll_steps', type=int, default=-1)
    parser.add_argument('--rl_train_steps', type=int, default=0)
    parser.add_argument('--exact_comp', action='store_true')
    parser.add_argument('--log_state_reprs', action='store_true')
    parser.add_argument('--tags', nargs='*', default=None)
    parser.add_argument('--extra_buffer_keys', nargs='*', default=[])
    parser.add_argument('--eval_policies', nargs='*', type=str,
                        default=['random'], action='store')
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--eval_unroll_steps', type=int, default=20)
    parser.add_argument('--log_norms', action='store_true') # Log trans model norm data
    parser.add_argument('--ae_grad_clip', type=float, default=0)
    parser.add_argument('--e2e_loss', action='store_true')
    add_model_args(parser)
    parser.set_defaults(
        preprocess=False, wandb=False, load=True, unique_data=False,
        cache=True, upload_model=False, save=False, preload_data=False,
        exact_comp=False, comet_ml=False, log_state_reprs=False,
        log_norms=False, e2e_loss=False)
    return parser



def add_model_args(parser):
  parser.add_argument('--embedding_dim', type=int, default=64)
  parser.add_argument('--latent_dim', type=int, default=None)
  parser.add_argument('--filter_size', type=int, default=8)
  parser.add_argument('--codebook_size', type=int, default=16)
  parser.add_argument('--ae_model_hash', type=str, default=None)
  parser.add_argument('--trans_model_hash', type=str, default=None)
  parser.add_argument('--trans_hidden', type=int, default=256)
  parser.add_argument('--trans_depth', type=int, default=3)
  parser.add_argument('--stochastic', type=str, default='simple',
            choices=[None, 'simple', 'categorical'])
  
  # Only works with ae
  parser.add_argument('--fta_tiles', type=int, default=20,
    help='How many tiles to use in FTA')
  parser.add_argument('--fta_bound_low', type=float, default=-2,
    help='Upper bound for FTA range')
  parser.add_argument('--fta_bound_high', type=float, default=2,
    help='Lower bound for FTA range')
  parser.add_argument('--fta_eta', type=float, default=0.2,
    help='Degree of fuzzyness in FTA')

  # Only works with soft_vqvae
  parser.add_argument('--repr_sparsity', type=float, default=0,
    help='Fractional sparsity of representations post training')
  parser.add_argument('--sparsity_type', type=str, default='random',
    choices=['random', 'identity'],
    help='Type of sparsity mask to use')
  
  # Only works with universal_vq transition model
  parser.add_argument('--vq_trans_loss_type', type=str, default='mse',
    choices=['mse', 'cross_entropy'])
  parser.add_argument('--vq_trans_1d_conv', action='store_true')
  parser.add_argument('--vq_trans_state_snap', action='store_true')

  parser.set_defaults(vq_trans_1d_conv=False, vq_trans_state_snap=False)



def validate_args(args):
    pass
    # if args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
    #     if args.trans_model_type != 'continuous':
    #         print(f'Model type {args.ae_model_type} requires a continuous transition model!')
    #         print('Ending run.')
    #         sys.exit()
    # elif args.ae_model_type in DISCRETE_ENCODER_TYPES:
    #     if args.trans_model_type not in ('discrete', 'transformer', 'transformerdec'):
    #         print(f'Model type {args.ae_model_type} requires a discrete transition model!')
    #         print('Ending run.')
    #         sys.exit()
    # elif args.ae_model_type != 'flatten':
    #     raise ValueError(f'Invalid encoder type, "{args.ae_model_type}"!')

    # if 'transformer' in args.trans_model_type.lower() and \
    #    args.stochastic and args.stochastic.lower() == 'categorical':
    #     raise ValueError('Transformer type transition models cannot use categorical stochasticity type!')

def log_param_updates(args, params):
  if args.wandb:
    import wandb
    if not isinstance(wandb.config, wandb.sdk.lib.preinit.PreInitObject):
        wandb.config.update(params, allow_val_change=True)
  elif args.comet_ml:
    import comet_ml
    experiment = comet_ml.get_global_experiment()
    if experiment is not None:
        experiment.log_parameters(params)

def update_arg(args, key, val):
    if args.wandb and not isinstance(args, Namespace):
        args._items.update({key: val})
    else:
        setattr(args, key, val)
    
    log_param_updates(args, {key: val})

def process_args(args):
    new_env_name = check_env_name(args.env_name)
    update_arg(args, 'env_name', new_env_name)
    if isinstance(args.eval_policies, str):
        update_arg(args, 'eval_policies', [args.eval_policies])

    if args.trans_learning_rate is None:
        update_arg(args, 'trans_learning_rate', args.learning_rate)

    validate_args(args)
    return args

def get_args(parser=None):
    parser = make_argparser(parser)
    args = parser.parse_args()
    return process_args(args)

def test_model(model, test_func, data_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_data in data_loader:
            loss = test_func(batch_data)
            if not isinstance(loss, dict):
                loss = {'loss': loss.mean()}
            losses.append(loss)
    model.train()
    return losses

def train_loop(model, trainer, train_loader, valid_loader=None, n_epochs=1,
               batch_size=128, log_freq=100, seed=0, callback=None,
               valid_callback=None, test_func=None):
    torch.manual_seed(seed)
    model.train()
    train_losses = []
    for epoch in range(n_epochs):
        print(f'Starting epoch #{epoch}')
        print('Memory usage: {:.1f} GB'.format(
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))
        for i, batch_data in enumerate(train_loader):
            train_loss, aux_data = trainer.train(batch_data)
            if not isinstance(train_loss, dict):
                train_loss = {'loss': train_loss}
            train_losses.append(train_loss)
            if callback:
                callback(train_loss, i*batch_size, epoch, aux_data=aux_data)
            if i % log_freq == 0:
                train_loss_means = {k: np.mean([x[k].item() for x in train_losses])
                                    for k in train_losses[0]}
                train_losses = []
                
                update_str = f'Epoch {epoch} | Samples {i * batch_size}'
                for k, v in train_loss_means.items():
                    update_str += f' | train_{k}: {v:.3f}'

                if valid_loader is not None:
                    test_func = test_func or trainer.calculate_losses
                    valid_losses = test_model(
                        model, test_func, valid_loader)
                    valid_loss_means = {k: np.mean([x[k].item() for x in valid_losses])
                                        for k in valid_losses[0]}
                    if valid_callback:
                        valid_callback(valid_loss_means, i, epoch)
                    for k, v in valid_loss_means.items():
                        update_str += f' | valid_{k}: {v:.3f}'
                    model.train()
                print(update_str)

def sample_recon_imgs(model, dataloader, n=4, env_name=None, rev_transform=None):
    # Generate sample reconstructions
    model.eval()
    
    # Check if dataloader has iter method
    if isinstance(dataloader, DataLoader):
        samples = next(iter(dataloader))[0][:n]
    elif isinstance(dataloader, torch.Tensor):
        samples = dataloader[:n]
    elif isinstance(dataloader, (list, tuple)):
        samples = torch.stack(dataloader[:n])
    else:
        raise ValueError(f'Invalid dataloader type: {type(dataloader)}!')

    device = next(model.parameters()).device
    with torch.no_grad():
        encs = model.encode(samples.to(device))
        decs = model.decode(encs)

    # Covert states to images
    samples = states_to_imgs(samples, env_name, transform=rev_transform)
    samples = torch.from_numpy(samples)

    decs = states_to_imgs(decs, env_name, transform=rev_transform)
    decs = torch.from_numpy(decs)
    
    sample_recons = torch.cat([samples, decs.cpu()], dim=3)
    sample_recons = sample_recons.clip(0, 1)
    if samples.shape[1] == 1 or samples.shape[1] == 3:
        sample_recons = sample_recons.permute(0, 2, 3, 1).numpy()
    else:
        srs = sample_recons.shape
        sample_recons = sample_recons.reshape(srs[0] * srs[1], *srs[2:]).numpy()
    return sample_recons

def sample_recon_seqs(encoder, trans_model, dataloader, n_steps, n=4,
                      env_name=None, rev_transform=None, gif_format=False):
    # Generate sample reconstructions
    encoder.eval()
    trans_model.eval()
    device = next(encoder.parameters()).device
    sample_batch = next(iter(dataloader))
    sample_batch = [x[:n].to(device) for x in sample_batch]
    if n_steps <= 1:
        # Add dummy dimension for single step prediction
        sample_batch = [x[:, None] for x in sample_batch]
    obs, acts, next_obs = sample_batch[:3]
    init_obs = obs[:, 0] # First step of each sequence
    z = encoder.encode(init_obs)

    dec_steps = [torch.zeros_like(init_obs)]
    for i in range(n_steps):
        with torch.no_grad():
            z = trans_model(z, acts[:, i])[0]
            decoded = encoder.decode(z)
        dec_steps.append(decoded)

    all_obs = torch.cat([obs[:, :1], next_obs], dim=1)
    dec_steps = torch.stack(dec_steps)

    # Covert states to images
    orig_shape = dec_steps.shape[:2]
    flat_dec_steps = dec_steps.reshape(-1, *dec_steps.shape[2:])
    flat_dec_steps = states_to_imgs(flat_dec_steps, env_name, transform=rev_transform)
    flat_dec_steps = torch.from_numpy(flat_dec_steps)
    dec_steps = flat_dec_steps.reshape(*orig_shape, *flat_dec_steps.shape[1:])
    dec_steps = list(dec_steps)

    orig_shape = all_obs.shape[:2]
    flat_all_obs = all_obs.reshape(-1, *all_obs.shape[2:])
    flat_all_obs = states_to_imgs(flat_all_obs, env_name, transform=rev_transform)
    flat_all_obs = torch.from_numpy(flat_all_obs)
    all_obs = flat_all_obs.reshape(*orig_shape, *flat_all_obs.shape[1:])
    
    if gif_format:
        sample_recons = torch.stack(dec_steps).transpose(0, 1)
        sample_recons = torch.cat([all_obs, sample_recons], dim=4).cpu()
        sample_recons = (sample_recons.numpy().clip(0, 1) * 255).astype(np.uint8)
        
        # Check for framestack, shape is (b, n, c, w, h)
        if sample_recons.shape[2] not in (1, 3):
            sample_recons = sample_recons[:, :, -1:] # Take last frame of each framestack
    else:
        # Spread n_steps out over width, (b, c, h, n * w)
        sample_recons = torch.cat(dec_steps, dim=3)
        all_obs = rearrange(all_obs, 'b n c h w -> b c h (n w)')

        # Stack real and predicted obs on top of each other
        sample_recons = torch.cat([all_obs, sample_recons], dim=2).cpu()
        sample_recons = sample_recons.permute(0, 2, 3, 1).numpy()
        sample_recons = sample_recons.clip(0, 1)
        
        # Check for framestack
        if sample_recons.shape[-1] not in (1, 3):
            sample_recons = sample_recons[..., -1:] # Take last frame of each framestack

    return sample_recons

def vec_env_random_walk(env, n_steps, progress=True):
    """ Generate random walks for n_steps. """
    env.reset()
    prog_func = tqdm if progress else lambda x: x
    for _ in prog_func(range(int(np.ceil(n_steps / env.num_envs)))):
        acts = [env.action_space.sample() for _ in range(env.num_envs)]
        env.step(acts)

def vec_env_ez_explore(env, n_steps, min_repeat=1, max_repeat=8, progress=True):
    """ Generate ez-explore walks for n_steps. """
    env.reset()

    curr_acts = np.array([0 for _ in range(env.num_envs)])
    curr_repeats = np.array([0 for _ in range(env.num_envs)])

    prog_func = tqdm if progress else lambda x: x
    for _ in prog_func(range(int(np.ceil(n_steps / env.num_envs)))):
        for i in range(env.num_envs):
            if curr_repeats[i] == 0:
                curr_acts[i] = env.action_space.sample()
                curr_repeats[i] = np.random.randint(min_repeat, max_repeat + 1)
        acts = [env.action_space.sample() for _ in range(env.num_envs)]
        env.step(acts)
        curr_repeats -= 1
