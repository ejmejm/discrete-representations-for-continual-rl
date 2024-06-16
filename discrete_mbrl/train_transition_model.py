from collections import defaultdict
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import functools

from shared.models import *
from shared.trainers import *
from data_helpers import *
from data_logging import *
from env_helpers import *
from training_helpers import *
from model_construction import *


TRANS_STEP = 0
train_log_buffer = defaultdict(list)
aux_log_buffer = defaultdict(list)


def train_trans_model(args, encoder_model=None):
    import_logger(args)
        
    print('Loading data...')
    # Data shape: (5, batch_size, n_steps, ...)
    train_loader, test_loader, valid_loader = prepare_dataloaders(
        args.env_name, n=args.max_transitions, batch_size=args.batch_size,
        n_step=args.n_train_unroll, preprocess=args.preprocess, randomize=True,
        n_preload=args.n_preload, preload_all=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys)

    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{len(valid_loader.dataset)}')

    if encoder_model is None:
        print('Constructing encoder...')
        sample_obs = next(iter(train_loader))[0][0]
        if args.n_train_unroll > 1:
            sample_obs = sample_obs[0]
        encoder_model = construct_ae_model(
            sample_obs.shape, args)[0]
    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()
    
    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    print('Constructing transition model...')
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    trans_model, trans_trainer = construct_trans_model(
        encoder_model, args, env.action_space, load=args.load)
    print('Transition model:', trans_model)
    trans_model = trans_model.to(args.device)
    test_func = functools.partial(
        trans_trainer.calculate_losses, n=args.n_train_unroll)
    trans_trainer.train = functools.partial(trans_trainer.train, n=args.n_train_unroll)

    update_params(args)
    track_model(trans_model, args)

    def train_callback(train_data, batch_idx, epoch, aux_data=None):
        global TRANS_STEP, train_log_buffer, aux_log_buffer
        if args.save and epoch % args.checkpoint_freq == 0 and batch_idx == 0:
            save_model(trans_model, args, model_hash=args.trans_model_hash)
            
        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            train_log_buffer[k].append(v)
        if aux_data is not None:
            for k, v in aux_data.items():
                aux_log_buffer[k].append(v)

        if TRANS_STEP % max(1, (args.log_freq // 10)) == 0:
            log_metrics({
                'epoch': epoch,
                'step': TRANS_STEP,
                **{f'train_{k}': sum(v) / len(v) for k, v in train_log_buffer.items()},
                **{k: sum(v) / len(v) for k, v in aux_log_buffer.items()}},
                args, prefix='trans', step=TRANS_STEP)
            train_log_buffer = defaultdict(list)
        TRANS_STEP += 1

    # For reversing observation transformations
    rev_transform = valid_loader.dataset.flat_rev_obs_transform
    
    def valid_callback(valid_data, batch_idx, epoch):
        global TRANS_STEP
        log_metrics({
            'epoch': epoch,
            'step': TRANS_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}},
            args, prefix='trans', step=TRANS_STEP)

        if batch_idx == 0 and epoch % args.checkpoint_freq == 0:
            valid_recons = sample_recon_seqs(
                encoder_model, trans_model, valid_loader, args.n_train_unroll,
                env_name=args.env_name, rev_transform=rev_transform, gif_format=True)
            train_recons = sample_recon_seqs(
                encoder_model, trans_model, train_loader, args.n_train_unroll,
                env_name=args.env_name, rev_transform=rev_transform, gif_format=True)
            log_videos({
                'valid_seq_recon': valid_recons,
                'train_seq_recon': train_recons},
                args, prefix='trans', step=TRANS_STEP)

    n_epochs = args.trans_epochs if args.trans_epochs is not None else args.epochs
    try:
        train_loop(
            trans_model, trans_trainer, train_loader, valid_loader, n_epochs,
            args.batch_size, args.log_freq, callback=train_callback,
            valid_callback=valid_callback, test_func=test_func)
    except KeyboardInterrupt:
        print('Stopping training')

    # Get rid of any remaining log data
    global train_log_buffer
    del train_log_buffer
    
    # Test the model
    print('Starting model evaluation...')
    test_losses = test_model(trans_model, test_func, test_loader)
    test_losses = {k: np.mean([d[k].item() for d in test_losses]) for k in test_losses[0].keys()}
    print(f'Transition model test losses: {test_losses}')

    if args.save:
        save_model(trans_model, args, model_hash=args.trans_model_hash)
        print('Transition model saved')

    return trans_model

if __name__ == '__main__':
    # Parse args
    args = get_args()
    # Setup logging
    args = init_experiment('discrete-mbrl-transition-models', args)
    # Train and test the model
    trans_model = train_trans_model(args)
    # Save the model
    if args.save:
        save_model(trans_model, args, model_hash=args.trans_model_hash)
        print('Model saved')