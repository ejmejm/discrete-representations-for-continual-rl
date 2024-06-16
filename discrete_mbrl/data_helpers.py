import gzip
import hashlib
import os
import pickle
import sys
import time
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from tqdm import tqdm

from env_helpers import DATA_DIR, MUJOCO_ENVS
from utils import to_hashable_tensor_list


MAX_TEST = 50_000
MAX_VALID = 5000
CACHE_DIR = 'data/cache/'


def construct_cache_path(env_name, preprocess, randomize, seed, n):
    name = DATA_DIR + '/' + env_name
    name += '_' + str(preprocess)
    name += '_' + str(randomize)
    name += '_' + str(seed)
    if n is not None:
        name += '_' + str(n)
    name += '.pkl.gz'
    return name

def load_cache(env_name, preprocess, randomize, seed, n):
    cache_path = construct_cache_path(env_name, preprocess, randomize, seed, n)
    with gzip.open(cache_path, 'rb') as f:
        return pickle.load(f)

def save_cache(env_name, preprocess, randomize, seed, n, cache):
    cache_path = construct_cache_path(env_name, preprocess, randomize, seed, n)
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


class Subset(Dataset):
    """
    Custom Subset that allows accessing the original dataset more easily.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith('_'):
            raise AttributeError(f'accessing private attribute "{name}" is prohibited')
        return getattr(self.dataset, name)

class NStepReplayDataset(Dataset):
    def __init__(self, env_name, n_steps, transform=None, cross_chunks=True, preload=False,
                 start_idx=0, end_idx=None, extra_buffer_keys=None):
        sanitized_env_name = env_name.replace(':', '_')
        self.replay_buffer_path = f'{DATA_DIR}/{sanitized_env_name}_replay_buffer.hdf5'
        self.n_steps = n_steps
        self.transform = transform
        self.cross_chunks = cross_chunks
        self.preload = preload
        with h5py.File(self.replay_buffer_path, 'r') as buffer:
            self.act_type = buffer.get('action').dtype
            self.act_type = torch.float32 if 'float' in str(self.act_type) else torch.int64
            self.chunk_size = buffer['obs'].chunks[0]
            self.data_keys = list(sorted(buffer.keys()))
            self.extra_keys = set(self.data_keys) - \
                set(['obs', 'action', 'next_obs', 'reward', 'done'])
            self.extra_keys = self.extra_keys.intersection(set(extra_buffer_keys))
            unused_keys = self.extra_keys - set(extra_buffer_keys)
            if len(unused_keys) > 0:
                print(f'Warning: unused keys in replay buffer: {unused_keys}')
                
            if buffer.attrs.get('obs_mean') is not None:
                self.flat_obs_mean = torch.from_numpy(buffer.attrs['obs_mean'])
                self.obs_mean = self.flat_obs_mean.unsqueeze(0)
                self.flat_obs_std = torch.from_numpy(buffer.attrs['obs_std'])
                self.obs_std = self.flat_obs_std.unsqueeze(0)
                
                self.obs_transform = lambda o: (o - self.obs_mean) / self.obs_std
                self.flat_obs_transform = lambda o: (o - self.flat_obs_mean) / self.flat_obs_std
                self.rev_obs_transform = lambda o: o * self.obs_std + self.obs_mean
                self.flat_rev_obs_transform = lambda o: o * self.flat_obs_std + self.flat_obs_mean
            else:
                self.obs_transform = lambda o: o
                self.flat_obs_transform = lambda o: o
                self.rev_obs_transform = lambda o: o
                self.flat_rev_obs_transform = lambda o: o

            self.extra_keys = list(sorted(self.extra_keys))

            if preload:
                if end_idx is None:
                    ei = buffer['obs'].shape[0]
                else:
                    ei = end_idx + self.n_steps - 1
                self.data_buffer = {}
                self.data_buffer['obs'] = torch.from_numpy(
                    buffer['obs'][start_idx:ei]).float()
                self.data_buffer['action'] = torch.tensor(
                    buffer['action'][start_idx:ei]).to(self.act_type)
                self.data_buffer['next_obs'] = torch.from_numpy(
                    buffer['next_obs'][start_idx:ei]).float()
                self.data_buffer['reward'] = torch.tensor(
                    buffer['reward'][start_idx:ei]).float()
                self.data_buffer['done'] = torch.tensor(
                    buffer['done'][start_idx:ei]).float()
                
                for key in self.extra_keys:
                    self.data_buffer[key] = torch.from_numpy(
                        buffer[key][start_idx:ei])

                self.n_samples = self.data_buffer['obs'].shape[0]
            else:
                self.n_samples = buffer.attrs['data_idx']

        if cross_chunks:
            self.length = self.n_samples - self.n_steps + 1
        else:
            # Calculate number of n_step samples that don't cross chunk boundaries
            n_full_chunks = self.chunk_size // self.n_samples
            samples_per_chunk = self.chunk_size - self.n_steps + 1
            extra_samples = max(0,
                (self.n_samples % self.chunk_size) - self.n_steps + 1)
            self.length = n_full_chunks * samples_per_chunk + extra_samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.cross_chunks:
            return self.__getitem_cross_chunks(idx)
        return self.__getitem_no_cross_chunks(idx)

    def __getitem_cross_chunks(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        if isinstance(idx, (slice, range)):
            idx = list(range(idx.start, idx.stop, idx.step))
        sample_idxs = []
        for i in idx:
            sample_idxs.extend(range(i, i + self.n_steps))
        if self.preload:
            obs = self.obs_transform(self.data_buffer['obs'][sample_idxs])
            action = self.data_buffer['action'][sample_idxs]
            next_obs = self.obs_transform(self.data_buffer['next_obs'][sample_idxs])
            reward = self.data_buffer['reward'][sample_idxs]
            done = self.data_buffer['done'][sample_idxs]
            extra_data = [self.data_buffer[key][sample_idxs] \
                for key in self.extra_keys]
        else:
            with h5py.File(self.replay_buffer_path, 'r') as buffer:
                obs = self.obs_transform(torch.from_numpy(buffer['obs'][sample_idxs]).float())
                action = torch.tensor(buffer['action'][sample_idxs]).to(self.act_type)
                next_obs = self.obs_transform(torch.from_numpy(buffer['next_obs'][sample_idxs]).float())
                reward = torch.tensor(buffer['reward'][sample_idxs]).float()
                done = torch.tensor(buffer['done'][sample_idxs]).float()
                extra_data = [torch.tensor(buffer[key][sample_idxs]) \
                    for key in self.extra_keys]

        transition_set = [obs, action, next_obs, reward, done, *extra_data]

        if self.transform:
            transition_set = \
                self.transform(*transition_set)

        transition_set = [x.reshape(int(x.shape[0] / self.n_steps), self.n_steps, *x.shape[1:]) \
            .squeeze(0) for x in transition_set]

        return transition_set

    def __getitem_no_cross_chunks(self, idx):
        raise NotImplementedError

class ReplayDataset(Dataset):
    def __init__(self, env_name, transform=None, preload=False,
                 start_idx=0, end_idx=None, extra_buffer_keys=None):
        sanitized_env_name = env_name.replace(':', '_')
        extra_buffer_keys = extra_buffer_keys or []
        self.replay_buffer_path = f'{DATA_DIR}/{sanitized_env_name}_replay_buffer.hdf5'
        self.transform = transform
        self.preload = preload
        # TODO: Load in act type dynamically, and change for nstep too, self.act_type 
        with h5py.File(self.replay_buffer_path, 'r') as buffer:
            self.act_type = buffer.get('action').dtype
            self.act_type = torch.float32 if 'float' in str(self.act_type) else torch.int64
            self.data_keys = list(sorted(buffer.keys()))
            self.extra_keys = set(self.data_keys) - \
                set(['obs', 'action', 'next_obs', 'reward', 'done'])
            self.extra_keys = self.extra_keys.intersection(set(extra_buffer_keys))
            unused_keys = self.extra_keys - set(extra_buffer_keys)
            if len(unused_keys) > 0:
                print(f'Warning: unused keys in replay buffer: {unused_keys}')
                
            if buffer.attrs.get('obs_mean') is not None:
                self.flat_obs_mean = torch.from_numpy(buffer.attrs['obs_mean'])
                self.obs_mean = self.flat_obs_mean.unsqueeze(0)
                self.flat_obs_std = torch.from_numpy(buffer.attrs['obs_std'])
                self.obs_std = self.flat_obs_std.unsqueeze(0)
                
                self.obs_transform = lambda o: (o - self.obs_mean) / self.obs_std
                self.flat_obs_transform = lambda o: (o - self.flat_obs_mean) / self.flat_obs_std
                self.rev_obs_transform = lambda o: o * self.obs_std + self.obs_mean
                self.flat_rev_obs_transform = lambda o: o * self.flat_obs_std + self.flat_obs_mean
            else:
                self.obs_transform = lambda o: o
                self.flat_obs_transform = lambda o: o
                self.rev_obs_transform = lambda o: o
                self.flat_rev_obs_transform = lambda o: o

            self.extra_keys = list(sorted(self.extra_keys))

            if preload:
                if end_idx is None:
                    end_idx = buffer['obs'].shape[0]
                self.data_buffer = {}
                self.data_buffer['obs'] = torch.from_numpy(
                    buffer['obs'][start_idx:end_idx]).float()
                self.data_buffer['action'] = torch.tensor(
                    buffer['action'][start_idx:end_idx]).to(self.act_type)
                self.data_buffer['next_obs'] = torch.from_numpy(
                    buffer['next_obs'][start_idx:end_idx]).float()
                self.data_buffer['reward'] = torch.tensor(
                    buffer['reward'][start_idx:end_idx]).float()
                self.data_buffer['done'] = torch.tensor(
                    buffer['done'][start_idx:end_idx]).float()

                for key in self.extra_keys:
                    self.data_buffer[key] = torch.from_numpy(
                        buffer[key][start_idx:end_idx])

                self.length = self.data_buffer['obs'].shape[0]
            else:
                self.length = buffer.attrs['data_idx']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = list(range(idx.start, idx.stop, idx.step or 1))
        elif isinstance(idx, range):
            idx = list(idx)
        elif isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        if self.preload:
            obs = self.data_buffer['obs'][idx]
            action = self.data_buffer['action'][idx]
            next_obs = self.data_buffer['next_obs'][idx]
            reward = self.data_buffer['reward'][idx]
            done = self.data_buffer['done'][idx]
            extra_data = [self.data_buffer[key][idx] \
                for key in self.extra_keys]
        else:
            with h5py.File(self.replay_buffer_path, 'r') as buffer:
                obs = torch.from_numpy(buffer['obs'][idx]).float()
                action = torch.tensor(buffer['action'][idx]).to(self.act_type)
                next_obs = torch.from_numpy(buffer['next_obs'][idx]).float()
                reward = torch.tensor(buffer['reward'][idx]).float()
                done = torch.tensor(buffer['done'][idx]).float()
                extra_data = [torch.tensor(buffer[key][idx]) \
                    for key in self.extra_keys]

        if isinstance(idx, list):
            obs = self.obs_transform(obs)
            next_obs = self.obs_transform(next_obs)
        else:
            obs = self.flat_obs_transform(obs)
            next_obs = self.flat_obs_transform(next_obs)

        if self.transform:
            obs, action, next_obs, reward, done = \
                self.transform(obs, action, next_obs, reward, done)

        return obs, action, next_obs, reward, done, *extra_data

# Source: https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
class NStepWeakBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, n_steps, shuffle=False):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.block_size = self.n_steps * self.batch_size
        self.dataset_length = len(dataset)
        self.n_batches = int(np.ceil(self.dataset_length / self.block_size)) * self.n_steps
        self.shuffle = shuffle
        self.batch_ids = torch.arange(self.n_batches)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(self.n_batches)
        for id in self.batch_ids:
            block_idx = id // self.n_steps
            idx = slice(block_idx * self.block_size + id % self.n_steps,
                min((block_idx + 1) * self.block_size, self.dataset_length), self.n_steps)
            yield idx
            
class WeakBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = int(np.ceil(self.dataset_length / self.batch_size))
        self.shuffle = shuffle
        self.batch_ids = torch.arange(self.n_batches)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(self.n_batches)
        for id in self.batch_ids:
            idx_slice = slice(id * self.batch_size, min(
                (id + 1) * self.batch_size, self.dataset_length))
            yield idx_slice

class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.shuffle = shuffle
        self.batch_ids = torch.randperm(self.dataset_length)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(self.dataset_length)
        for i in range(0, self.dataset_length, self.batch_size):
            idxs = self.batch_ids[i:i+self.batch_size]
            idxs = torch.sort(idxs)[0]
            yield idxs

def preprocess_transform(obs, action, next_obs, reward, done):
    obs[0] /= 255.0
    next_obs /= 255.0
    # Check if the channel is already in the right dimension
    permute = min(obs[0].shape) != obs[0].shape[0]
    if permute:
        obs = obs.permute(0, 3, 1, 2)
        next_obs = next_obs.permute(0, 3, 1, 2)
    return obs, action, next_obs, reward, done


TEST_FRAC = 0.1
VALID_FRAC = 0.01

def create_fast_loader(
        dataset, batch_size, shuffle=False, num_workers=0,
        n_step=1, weak_shuffle=False):
    sampler = None
    if weak_shuffle:
        sampler = WeakBatchSampler(dataset, batch_size, shuffle) if n_step <= 1 \
            else NStepWeakBatchSampler(dataset, batch_size, n_step, shuffle)
        return DataLoader(
            dataset, num_workers=num_workers, collate_fn=lambda x: x[0], sampler=sampler)

    return DataLoader(
        dataset, num_workers=num_workers, drop_last=False,
        shuffle=shuffle, batch_size=batch_size)

def prepare_dataloaders(env_name, batch_size=256, randomize=True, n_step=1,
                        preprocess=False, n=None, n_preload=0, seed=None,
                        valid_preload=True, preload_all=False, extra_buffer_keys=None):
    transform = preprocess_transform if preprocess else None
    if n_step > 1:
        dataset = NStepReplayDataset(
            env_name, n_step, transform=transform, preload=preload_all,
            extra_buffer_keys=extra_buffer_keys)
    else:
        dataset = ReplayDataset(
            env_name, transform, preload=preload_all,
            extra_buffer_keys=extra_buffer_keys)
    if n and n < len(dataset):
        dataset.length = min(dataset.length, n)

    n = len(dataset)
    n_test = min(int(n * TEST_FRAC), MAX_TEST)
    n_valid = min(int(n * VALID_FRAC), MAX_VALID)
    n_train = n - n_test - n_valid
    train_dataset = Subset(dataset, torch.arange(n_train))
    test_dataset = Subset(dataset, torch.arange(n_train, n-n_valid))

    if valid_preload and not preload_all:
        if n_step > 1:
            valid_dataset = NStepReplayDataset(env_name, n_step, transform=transform,
                preload=True, start_idx=n-n_valid, end_idx=n, extra_buffer_keys=extra_buffer_keys)
        else:
            valid_dataset = ReplayDataset(env_name, transform, preload=True,
                start_idx=n-n_valid, end_idx=n, extra_buffer_keys=extra_buffer_keys)
    else:
        valid_dataset = Subset(dataset, torch.arange(n-n_valid, n))

    if seed is not None and randomize:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    weak_shuffle = not preload_all
    valid_weak_shuffle = not valid_preload and not preload_all
    train_loader = create_fast_loader(
        train_dataset, batch_size, randomize, n_preload, n_step, weak_shuffle)
    test_loader = create_fast_loader(
        test_dataset, batch_size, False, n_preload, n_step, weak_shuffle)
    valid_loader = create_fast_loader(
        valid_dataset, batch_size, False, 0, n_step, valid_weak_shuffle)

    return train_loader, test_loader, valid_loader

def prepare_unique_obs_dataloader(args, randomize=True, preprocess=False, seed=None):
    if seed is not None and randomize:
        torch.manual_seed(seed)
        np.random.seed(seed)

    transform = preprocess_transform if preprocess else None
    unique_obs = get_unique_obs(args, cache=True, partition='all')
    dataset = TensorDataset(unique_obs)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=randomize, drop_last=False)

    return dataloader

def prepare_dataloader(env_name, partition, batch_size=256, randomize=True, n_step=1,
                       preprocess=False, n=None, n_preload=0, preload=False, seed=None,
                       extra_buffer_keys=None):
    transform = preprocess_transform if preprocess else None
    if n_step > 1:
        dataset = NStepReplayDataset(
            env_name, n_step, transform=transform,
            extra_buffer_keys=extra_buffer_keys)
    else:
        dataset = ReplayDataset(env_name, transform, extra_buffer_keys=extra_buffer_keys)
    if n and n < len(dataset):
        dataset.length = min(dataset.length, n)

    n = len(dataset)
    n_test = min(int(n * TEST_FRAC), MAX_TEST)
    n_valid = min(int(n * VALID_FRAC), MAX_VALID)
    n_train = n - n_test - n_valid

    partition_map = {
        'all': (0, n),
        'train': (0, n_train),
        'test': (n_train, n-n_valid),
        'valid': (n-n_valid, n),
    }
    start_idx, end_idx = partition_map[partition]

    if n_step > 1:
        dataset = NStepReplayDataset(
            env_name, n_step, transform=transform, preload=preload,
            start_idx=start_idx, end_idx=end_idx, extra_buffer_keys=extra_buffer_keys)
    else:
        if preload:
            dataset = ReplayDataset(env_name, transform, preload=True,
                start_idx=start_idx, end_idx=end_idx, extra_buffer_keys=extra_buffer_keys)
        else:
            dataset = Subset(dataset, np.arange(start_idx, end_idx))

    if seed is not None and randomize:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    weak_shuffle = not preload
    dataloader = create_fast_loader(
        dataset, batch_size, randomize, n_preload, n_step, weak_shuffle)

    return dataloader

def load_data_buffer(env_name, preprocess=True, randomize=True, seed=0,
                     n=None, cache=True):
    if cache:
        cache_path = construct_cache_path(env_name, preprocess, randomize, seed, n)
        if os.path.exists(cache_path):
            print('Found cache, loading...')
            return load_cache(env_name, preprocess, randomize, seed, n)
    else:
        print('Data caching disabled')

    # Load the replay buffer
    sanitized_env_name = env_name.replace(':', '_')
    replay_buffer_path = f'{DATA_DIR}/{sanitized_env_name}_replay_buffer.pkl.gz'
    with gzip.open(replay_buffer_path, 'rb') as f:
        replay_buffer = pickle.load(f)
    print('Replay buffer size:', len(replay_buffer),
        sys.getsizeof(replay_buffer[0]) * sys.getsizeof(replay_buffer))
    if n:
        replay_buffer = replay_buffer[:n]
        print('Truncated replay buffer size:', len(replay_buffer),
            sys.getsizeof(replay_buffer[0]) * sys.getsizeof(replay_buffer))

    print('Stacking data...')
    transition_data = [np.stack([x[i] for x in replay_buffer]) \
        for i in range(len(replay_buffer[0]))]
    transition_data = [torch.from_numpy(x).float() for x in transition_data]
    transition_data[1] = transition_data[1].long()
    del replay_buffer
    
    if preprocess:
        transition_data[0] = (transition_data[0] / 255)
        transition_data[2] = (transition_data[2] / 255)

        # Check if the channel is already in the right dimension
        permute = min(transition_data[0][0].shape) != transition_data[0][0].shape[0]
        if permute:
            transition_data[0] = transition_data[0].permute(0, 3, 1, 2)
            transition_data[2] = transition_data[2].permute(0, 3, 1, 2)

    if randomize:
        print('Randomizing data...')
        torch.manual_seed(seed)
        rand_idxs = torch.randperm(transition_data[0].shape[0])
        transition_data = [x[rand_idxs] for x in transition_data]

    if cache:
        print('Saving cache...')
        save_cache(env_name, preprocess, randomize, seed, n, transition_data)

    return transition_data

def get_md5(path, max_bytes=2**20, extra_data=None):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        md5.update(f.read(max_bytes)) # Only get the first 1MB
    if extra_data is not None:
        if isinstance(extra_data, Iterable):
            extra_data = ''.join([str(x) for x in extra_data])
        md5.update(str(extra_data).encode('utf-8'))

    return md5.hexdigest()

def get_unique_obs(args, cache=True, partition='all', early_stop_frac=1.0,
                   return_hash=False):
    if cache:
        replay_buffer_path = ReplayDataset(args.env_name).replay_buffer_path
        extra_data = []
        if args.max_transitions is not None:
            extra_data.append(args.max_transitions)
        if partition is not None:
            extra_data.append(partition)
        unique_data_hash = get_md5(
            replay_buffer_path, extra_data=extra_data)
        unique_data_path = os.path.join(CACHE_DIR, f'{unique_data_hash}.pkl')
        if os.path.exists(unique_data_path):
            print(f'Loading cached unique data from {unique_data_path}...')
            time_since_update = time.time() - os.path.getmtime(unique_data_path)
            if time_since_update < 60:
                time.sleep(max(0, 60 - time_since_update))
            with open(unique_data_path, 'rb') as f:
                unique_obs = pickle.load(f)
            print(f'{len(unique_obs)} unique observations loaded!')

            if return_hash:
                return unique_obs, unique_data_hash
            return unique_obs

    # If data was not already saved in cache, we need to compute it
    
    dataloader = prepare_dataloader(
        args.env_name, partition, batch_size=args.batch_size, preprocess=args.preprocess,
        randomize=True, n=args.max_transitions, n_preload=args.n_preload, preload=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys)

    dataset_size = len(dataloader.dataset)
    obs_since_update = 0
    unique_obs = set()

    print('Collecting unique observations...')
    for _, batch_data in enumerate(tqdm(dataloader)):
        obs, acts, next_obs = batch_data[:3]
        obs_list = to_hashable_tensor_list(obs)
        next_obs_list = to_hashable_tensor_list(next_obs)
        obs_list = obs_list + next_obs_list

        pre_len = len(unique_obs)
        new_obs_set = set(obs_list)
        unique_obs.update(new_obs_set)
        new_len = len(unique_obs)

        if pre_len == new_len:
            obs_since_update += 1
            if obs_since_update / dataset_size > early_stop_frac:
                print('Stopping unique obs collection early because no new obs were found!')
                break
        else:
            obs_since_update = 0

    unique_obs = torch.stack([x._tensor for x in unique_obs])
    print(f'{len(unique_obs)} Unique observations were gathered!')
            
    del dataloader

    # Save the unique obs if caching is enabled
    if cache:
        print(f'Saving unique data cache to {unique_data_path}...')
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(unique_data_path, 'wb') as f:
            pickle.dump(unique_obs, f)
        print('Saved!')

    if return_hash:
        return unique_obs, unique_data_hash
    return unique_obs