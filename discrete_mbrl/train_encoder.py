from collections import defaultdict
import os
import sys
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from shared.models import *
from shared.trainers import *
from data_helpers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from data_logging import *


ENCODER_STEP = 0
train_log_buffer = defaultdict(float)


def train_encoder(args):
    print('Loading data...')
    if args.unique_data:
        # In this case there is no valid/test data split
        train_loader = test_loader = \
            prepare_unique_obs_dataloader(args, randomize=True)
        valid_loader = None
    else:
        train_loader, test_loader, valid_loader = prepare_dataloaders(
            args.env_name, n=args.max_transitions, batch_size=args.batch_size,
            preprocess=args.preprocess, randomize=True, n_preload=args.n_preload,
            preload_all=args.preload_data, extra_buffer_keys=args.extra_buffer_keys)

    valid_len = len(valid_loader.dataset) if valid_loader is not None else 0
    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{valid_len}')

    print('Constructing model...')
    pre_sample_time = time.time()
    sample_obs = next(iter(train_loader))[0]
    print('Sample time:', time.time() - pre_sample_time)
    print('Sample shape:', sample_obs.shape)

    model, trainer = construct_ae_model(
        sample_obs.shape[1:], args, load=args.load)
    update_params(args)
    track_model(model, args)

    if hasattr(model, 'disable_sparsity'):
        model.disable_sparsity()

    if args.epochs <= 0:
        return model
    
    if trainer is not None:
        trainer.recon_loss_clip = args.recon_loss_clip
    model = model.to(args.device)
    print('# Params:', sum([x.numel() for x in model.parameters()]))
    print(model)

    def train_callback(train_data, batch_idx, epoch, **kwargs):
        global ENCODER_STEP, train_log_buffer
        if args.save and epoch % args.checkpoint_freq == 0 and batch_idx == 0:
            save_model(model, args, model_hash=args.ae_model_hash)

        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            train_log_buffer[k] += v
            train_log_buffer[f'{k}_count'] += 1

        if ENCODER_STEP % max(1, (args.log_freq // 10)) == 0:
            log_stats = {}
            for k, v in train_log_buffer.items():
                if k.endswith('_count'):
                    continue
                log_stats[k] = v / train_log_buffer[f'{k}_count']

            log_metrics({
                'epoch': epoch,
                'step': ENCODER_STEP,
                **log_stats},
                args, prefix='encoder', step=ENCODER_STEP)
            train_log_buffer = defaultdict(float)
        
        ENCODER_STEP += 1

    env = make_env(args.env_name, max_steps=args.env_max_steps)
    # For reversing observation transformations
    rev_transform = valid_loader.dataset.flat_rev_obs_transform
    
    def valid_callback(valid_data, batch_idx, epoch):
        global ENCODER_STEP
        log_metrics({
            'epoch': epoch,
            'step': ENCODER_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}},
            args, prefix='encoder', step=ENCODER_STEP)

        if batch_idx == 0 and epoch % args.checkpoint_freq == 0:
            valid_recons = sample_recon_imgs(
                model, valid_loader, env_name=args.env_name, rev_transform=rev_transform)
            train_recons = sample_recon_imgs(
                model, train_loader, env_name=args.env_name, rev_transform=rev_transform)
            log_images({
                'valid_img_recon': valid_recons,
                'train_img_recon': train_recons},
                args, prefix='encoder', step=ENCODER_STEP)

    try:
        train_loop(
            model, trainer, train_loader, valid_loader, args.epochs,
            args.batch_size, args.log_freq, callback=train_callback,
            valid_callback=valid_callback)
    except KeyboardInterrupt:
        print('Stopping training')

    # Get rid of any remaining log data
    global train_log_buffer
    del train_log_buffer

    # Test the model
    print('Starting model evaluation...')
    test_losses = test_model(model, trainer.calculate_losses, test_loader)
    test_losses = {k: np.mean([d[k].item() for d in test_losses]) for k in test_losses[0].keys()}
    print(f'Encoder test loss: {test_losses}')

    if args.save:
        save_model(model, args, model_hash=args.ae_model_hash)
        print('Encoder model saved')

    return model

if __name__ == '__main__':
    # Parse args
    args = get_args()
    # Setup logging
    args = init_experiment('discrete-mbrl-encoder', args)
    # Train and test the model
    model = train_encoder(args)
    # Save the model
    if args.save:
        save_model(model, args, model_hash=args.ae_model_hash)
        print('Model saved')
