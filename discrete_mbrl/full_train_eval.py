import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data_logging import init_experiment, finish_experiment
from env_helpers import *
from training_helpers import *
from train_encoder import train_encoder
from train_transition_model import train_trans_model
from evaluate_model import eval_model
from train_rl_model import train_rl_model
from e2e_train import full_train

def main():
    # Parse args
    args = get_args()

    # Setup logging
    args = init_experiment('discrete-mbrl-full', args)

    if args.e2e_loss:
        # Train and test end-to-end model
        encoder_model, trans_model = full_train(args)
    else:
        # Train and test the encoder model
        encoder_model = train_encoder(args)
        # Train and test the transition model
        trans_model = train_trans_model(args, encoder_model)
        # Train and evaluate an RL model with the learned model
        if args.rl_train_steps > 0:
            train_rl_model(args, encoder_model, trans_model)

    # Evalulate the models
    eval_model(args, encoder_model, trans_model)

    # Clean up logging
    finish_experiment(args)

if __name__ == '__main__':
    main()
    # from memory_profiler import memory_usage
    # mem_usage = memory_usage(main, interval=0.01)
    # print(f'Memory usage (in chunks of .1 seconds): {mem_usage}')
    # print(f'Maximum memory usage: {max(mem_usage)}')
