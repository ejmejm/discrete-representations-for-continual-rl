import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
from threading import Lock

from gym import spaces
import h5py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_helpers import *
from shared.models import SB3GeneralEncoder, SB3ActorCriticPolicy
from training_helpers import vec_env_random_walk, vec_env_ez_explore
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_name', type=str, default='MiniGrid-MultiRoom-N2-S4-v0')
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
parser.add_argument('-n', '--n_envs', type=int, default=8)
parser.add_argument('-s', '--train_steps', type=int, default=int(3e5))
parser.add_argument('-c', '--chunk_size', type=int, default=2048)
parser.add_argument('-ct', '--compression_type', type=str, default='lzf')
# ppo, ppo_entropy, random, ezexplore
parser.add_argument('-a', '--algorithm', type=str, default='random')
parser.add_argument('--extra_info', nargs='*', default=[])
parser.add_argument('--norm_stats', action='store_true')
parser.add_argument('--shrink_size', type=int, default=None,
    help='If given, the final dataset will be shrunken down to this size by \
          taking half from the start of training and the other half from the end.')
parser.set_defaults(norm_stats=False)


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, replay_buffer, buffer_lock):
        super(EarlyStoppingCallback, self).__init__(verbose=0)
        self.replay_buffer = replay_buffer
        self.lock = buffer_lock
        self.check_interval = 10000
        self.idx = 0

    def _on_step(self) -> bool:
        self.idx += 1
        if self.idx % self.check_interval != 0:
            return True

        with self.lock:
            n_samples = self.replay_buffer.attrs['data_idx']
            buffer_size = self.replay_buffer['obs'].shape[0]
            if n_samples >= buffer_size:
                print('Enough data collected, stopping training')
                return False
        print(f'{n_samples}/{buffer_size} transitions recorded')
        return True

def setup_replay_buffer(args, path=None):
    sanitized_env_name = args.env_name.replace(':', '_')
    replay_buffer_path = path or f'./data/{sanitized_env_name}_replay_buffer.hdf5'
    replay_buffer = h5py.File(replay_buffer_path, 'w')
    dataset_size = args.train_steps
    # dataset_size = args.shrink_size or args.train_steps
    # replay_buffer.capacity = dataset_size
    # replay_buffer.train_steps = args.train_steps

    env = make_env(args.env_name, max_steps=args.env_max_steps)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    act_type = 'int32' if isinstance(env.action_space, spaces.Discrete) else 'float32'

    replay_buffer.create_dataset('obs', (dataset_size, *obs_shape), maxshape=(args.train_steps, *obs_shape),
        compression=args.compression_type, dtype='float16', chunks=(args.chunk_size, *obs_shape))
    if len(act_shape) == 0:
        replay_buffer.create_dataset('action', (dataset_size,), maxshape=(args.train_steps,),
            compression=args.compression_type, dtype=act_type, chunks=(args.chunk_size,))
    else:
        replay_buffer.create_dataset('action', (dataset_size, *act_shape), maxshape=(args.train_steps, *act_shape),
            compression=args.compression_type, dtype=act_type, chunks=(args.chunk_size, *act_shape))
    replay_buffer.create_dataset('next_obs', (dataset_size, *obs_shape), maxshape=(args.train_steps, *obs_shape),
        compression=args.compression_type, dtype='float16', chunks=(args.chunk_size, *obs_shape))
    replay_buffer.create_dataset('reward', (dataset_size,), maxshape=(args.train_steps,),
        compression=args.compression_type, dtype='float32', chunks=(args.chunk_size,))
    replay_buffer.create_dataset('done', (dataset_size,), maxshape=(args.train_steps,),
        compression=args.compression_type, dtype='bool', chunks=(args.chunk_size,))
    replay_buffer.attrs['data_idx'] = 0
    
    if args.extra_info:
        env.reset()
        info = env.step(env.action_space.sample())[3]
        for key in args.extra_info:
            assert key in info, f'Key {key} not in info dict!'
            dtype = 'int16' if 'int' in str(info[key].dtype) else 'float16'
            print(f'Found key `{key}` with shape {info[key].shape} and dtype {dtype}')
            replay_buffer.create_dataset(key, (dataset_size, *info[key].shape),
                maxshape=(args.train_steps, *info[key].shape), compression=args.compression_type,
                dtype=dtype, chunks=(args.chunk_size, *info[key].shape))

    return replay_buffer

def shrink_replay_buffer(replay_buffer, new_size):
    half_size = int(new_size / 2)
    for key in replay_buffer.keys():
        # Make the second half of the buffer the newest data
        replay_buffer[key][new_size-half_size:new_size] = replay_buffer[key][-half_size:]
        if len(replay_buffer[key].shape) == 1:
            replay_buffer[key].resize((new_size,))
        else:
            replay_buffer[key].resize((new_size, *replay_buffer[key].shape[1:]))
    replay_buffer.attrs['data_idx'] = min(replay_buffer.attrs['data_idx'], new_size)

if __name__ == '__main__':
    args = parser.parse_args()
    args.env_name = check_env_name(args.env_name)

    # env = make_env(args.env_name)
    # obs = env.reset()
    # print(type(obs), obs.shape)
    # for _ in range(10):
    #     obs = env.step(env.action_space.sample())[0]
    #     print(type(obs), obs.shape, obs)
    # import matplotlib.pyplot as plt
    # print(obs.shape)
    # if len(obs.shape) == 4 or (len(obs.shape) == 3 and obs.shape[0] > 3):
    #     obs = obs[-1:]
    # if isinstance(obs, torch.Tensor):
    #     # plt.imshow(obs[-1])
    #     plt.imshow(obs[-3:].permute(1, 2, 0))
    # else:
    #     plt.imshow(obs.transpose(1, 2, 0))
    # plt.show()

    replay_buffer = setup_replay_buffer(args)
    buffer_lock = Lock()
    venv = DummyVecEnv([lambda: make_env(
        args.env_name, replay_buffer, buffer_lock,
        extra_info=args.extra_info, monitor=True,
        max_steps=args.env_max_steps)] * args.n_envs)

    if args.algorithm.lower().startswith('ppo'):
        policy_kwargs = dict(
        # Encoder
        features_extractor_class = SB3GeneralEncoder,
        features_extractor_kwargs = {
            'features_dim': 256,
            'fta': False},
        # Policy
        policy_fta = False,
        critic_fta = False,
        hidden_sizes = [256])
        entropy_coef = 0.01 if args.algorithm.lower().endswith('entropy') else 0.0
        model = PPO(SB3ActorCriticPolicy, venv, policy_kwargs=policy_kwargs,
                    verbose=1, tensorboard_log='mbrl_runs', learning_rate=args.learning_rate,
                    ent_coef=entropy_coef)
        # model = PPO('CnnPolicy', venv, verbose=1, tensorboard_log='mbrl_runs')
        model.learn(args.train_steps + 2048 * args.n_envs, log_interval=1,
            callback=EarlyStoppingCallback(replay_buffer, buffer_lock))
    elif args.algorithm == 'random':
        vec_env_random_walk(venv, args.train_steps)
    elif args.algorithm == 'ezexplore':
        vec_env_ez_explore(venv, args.train_steps)
    else:
        raise ValueError(f'Unknown algorithm: {args.algorithm}')

    if args.shrink_size and args.shrink_size < args.train_steps:
        print('-------')
        print(replay_buffer['obs'].shape)
        last_obs = replay_buffer['obs'][-1].copy()
        shrink_replay_buffer(replay_buffer, args.shrink_size)
        print(replay_buffer['obs'].shape)
        new_last_obs = replay_buffer['obs'][-1].copy()
        print((last_obs == new_last_obs).all())

    if args.norm_stats:
        print('Calculating normalization stats...')
        obs_count = replay_buffer.attrs['data_idx']
        obs_sum = np.zeros(replay_buffer['obs'].shape[1:], dtype=np.float64)
        obs_square_sum = np.zeros(replay_buffer['obs'].shape[1:], dtype=np.float64)
        for i in tqdm(range(0, args.train_steps, args.chunk_size)):
            obs_batch = replay_buffer['obs'][i:i+args.chunk_size]
            obs_sum += obs_batch.sum(axis=0)
            obs_square_sum += np.square(obs_batch).sum(axis=0)
            
        obs_mean = obs_sum / obs_count
        obs_std = np.sqrt((obs_count * obs_square_sum - np.square(obs_sum)) \
            / (obs_count * (obs_count - 1)))
        
        replay_buffer.attrs['obs_mean'] = obs_mean.astype(np.float32)
        replay_buffer.attrs['obs_std'] = obs_std.astype(np.float32)
        
    replay_buffer.close()