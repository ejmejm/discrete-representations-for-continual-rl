from collections import namedtuple
import os

import cv2
import numpy as np
import gym
from gym import Wrapper
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, ReseedWrapper
from gym_minigrid.wrappers import FullyObsWrapper, OBJECT_TO_IDX
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation, TransformReward 
from gym.wrappers.flatten_observation import FlattenObservation
from gym.core import ObservationWrapper
import torch
from stable_baselines3.common.monitor import Monitor

# Transition tuple
Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'reward', 'done'))

DATA_DIR = './data' # '/mnt/z/data'

# Wrapper for gym environments that stores trajectories and saves them to a buffer
class TrajectoryRecorderWrapper(Wrapper):
  def __init__(self, env, external_buffer, buffer_lock, extra_info=None):
    super().__init__(env)
    self.external_buffer = external_buffer
    self.lock = buffer_lock
    self.extra_info = extra_info or []
    self._reset_buffer()
    self.last_obs = None

  def _reset_buffer(self):
    self.ep_buffer = [[], [], [], [], []]
    for _ in range(len(self.extra_info)):
      self.ep_buffer.append([])

  def reset(self):
    self.last_obs = self.env.reset()
    return self.last_obs

  def update_external_buffer(self):
    with self.lock:
      curr_idx = self.external_buffer.attrs['data_idx']
      buffer_size = self.external_buffer['obs'].shape[0]
      if curr_idx >= buffer_size:
        return

      n_transitions = len(self.ep_buffer[0])
      if n_transitions <= 0:
        return
      end_idx = min(curr_idx + n_transitions, buffer_size)
      n_entries = end_idx - curr_idx
      
      self.external_buffer['obs'][curr_idx:end_idx] = \
        np.stack(self.ep_buffer[0][:n_entries], axis=0)
      self.external_buffer['action'][curr_idx:end_idx] = \
        np.stack(self.ep_buffer[1][:n_entries], axis=0)
      self.external_buffer['next_obs'][curr_idx:end_idx] = \
        np.stack(self.ep_buffer[2][:n_entries], axis=0)
      self.external_buffer['reward'][curr_idx:end_idx] = \
        np.stack(self.ep_buffer[3][:n_entries], axis=0)
      self.external_buffer['done'][curr_idx:end_idx] = \
        np.stack(self.ep_buffer[4][:n_entries], axis=0)
      
      for i, info_key in enumerate(self.extra_info):
        self.external_buffer[info_key][curr_idx:end_idx] = \
          np.stack(self.ep_buffer[5+i][:n_entries], axis=0)

      self.external_buffer.attrs['data_idx'] = end_idx

    self._reset_buffer()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    extra_info_vals = [info[key] for key in self.extra_info]

    for i, val in enumerate((self.last_obs, action, obs, reward,
                 done, *extra_info_vals)):
      self.ep_buffer[i].append(val)
    self.last_obs = obs
    if done:
      self.update_external_buffer()
    return obs, reward, done, info
  
  def close(self):
    self.update_external_buffer()
    return self.env.close()

class RescaleWrapper(ObservationWrapper):
  def __init__(self, env, low, high):
    super().__init__(env)
    self.orig_low = env.observation_space.low
    self.orig_high = env.observation_space.high
    self.orig_range = self.orig_high - self.orig_low
    orig_shape = env.observation_space.shape

    # Repeat dims to match observation shape
    for i, dim in enumerate(low.shape):
      if dim == 1 and orig_shape[i] > 1:
        low = low.repeat(orig_shape[i], axis=i)
        high = high.repeat(orig_shape[i], axis=i)

    self.observation_space = gym.spaces.Box(
      low=low, high=high, shape=env.observation_space.shape, dtype=np.float32)

  def observation(self, observation):
    return (observation - self.orig_low) / self.orig_range

class Custom2DWrapper(Wrapper):
  def __init__(
    self,
    env,
    frame_skip=1,
    rescale_size=None,
    grayscale_obs=False,
    grayscale_newaxis=True,
    scale_obs=True):
    super().__init__(env)
    assert frame_skip > 0

    self.frame_skip = frame_skip
    self.rescale = rescale_size is not None 
    self.rescale_size = rescale_size
    self.grayscale_obs = grayscale_obs
    self.grayscale_newaxis = grayscale_newaxis
    self.scale_obs = scale_obs

    _low, _high, _obs_dtype = (
      (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
    )
    
    if self.rescale:
      _shape = (1 if grayscale_obs else 3, rescale_size, rescale_size)
    else:
      _shape = (1 if grayscale_obs else 3, *self.observation_space.shape[:2])
    
    if grayscale_obs and not grayscale_newaxis:
      _shape = _shape[:-1]  # Remove channel axis
    self.observation_space = gym.spaces.Box(
      low=_low, high=_high, shape=_shape, dtype=_obs_dtype
    )

  def step(self, action):
    R = 0.0
    for _ in range(self.frame_skip):
      obs, reward, done, info = self.env.step(action)
      R += reward
      self.game_over = done
      if done:
        break
    return self.observation(obs), R, done, info

  def reset(self, **kwargs):
    # NoopReset
    obs = self.env.reset(**kwargs)
    return self.observation(obs)

  def observation(self, obs):
    # Resize the dimensions
    if self.rescale:
      obs = cv2.resize(obs, (self.rescale_size, self.rescale_size),
              interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    if self.grayscale_obs:
      obs = np.sum(obs * np.array([[[0.2989, 0.5870, 0.1140]]]), axis=2,
            keepdims=self.grayscale_newaxis)
    obs = obs.transpose(2, 0, 1)

    # Rescale obs to [0, 1]
    if self.scale_obs:
      obs = obs / 255
    
    return obs

class RenderWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

  def observation(self, observation):
    return self.render(mode='rgb_array')

class SqueezeDimWrapper(gym.ObservationWrapper):
  def __init__(self, env, dim=1):
    super().__init__(env)
    self.dim = dim
    if self.observation_space.shape[dim] == 1:
      self.observation_space = gym.spaces.Box(
        low=self.observation_space.low.squeeze(dim),
        high=self.observation_space.high.squeeze(dim),
        shape=self.observation_space.shape[:dim] + self.observation_space.shape[dim + 1:],
      )

  def observation(self, observation):
    return observation.squeeze(self.dim)

class MiniGridSimpleStochActionWrapper(Wrapper):
  def __init__(self, env, n_acts=None, stoch_probs=None):
    super().__init__(env)

    if n_acts is None:
      n_acts = env.action_space.n
    else:
      self.action_space = gym.spaces.Discrete(n_acts)

    self.act_idxs = list(np.arange(n_acts))
    target_act_prob = 0.9
    other_act_prob = (1 - target_act_prob) / (n_acts - 1)

    if stoch_probs is None:
      stoch_probs = np.eye(n_acts) * target_act_prob
      stoch_probs[stoch_probs == 0] = other_act_prob
    self.stoch_probs = stoch_probs

    self.stoch_enabled = True

  def action(self, action):
    if not self.stoch_enabled:
      return action
    return np.random.choice(self.act_idxs, p=self.stoch_probs[action])

  def step(self, action):
    true_action = self.action(action)
    obs, reward, done, info = self.env.step(true_action)
    info = {'true_action': true_action, **info}
    return obs, reward, done, info

  def enable_stochasticity(self):
    self.stoch_enabled = True

  def disable_stochasticity(self):
    self.stoch_enabled = False

def preprocess_obs(obs_list):
  return torch.from_numpy(np.stack(obs_list)).to(torch.float16).float()

def preprocess_act(act_list):
  return torch.from_numpy(np.stack(act_list)).long()

class PredictedModelWrapper(Wrapper):
  def __init__(self, env, encoder, trans_model):
    super().__init__(env)
    self.encoder = encoder
    self.trans_model = trans_model
    self.device = next(self.encoder.parameters()).device

    test_input = np.ones(env.observation_space.shape)
    test_input = preprocess_obs([test_input])
    with torch.no_grad():
      obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]
    self.observation_space = gym.spaces.Box(
      low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
    )

  def observation(self, obs):
    obs = preprocess_obs([obs])
    with torch.no_grad():
      obs = self.encoder.encode(obs.to(self.device))[0]
    return obs.cpu()

  def action(self, act):
    return preprocess_act([act])[0]

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self._curr_obs = self.observation(obs)
    return self._curr_obs

  def step(self, action):
    action = self.action(action)

    trans_out = self.trans_model(
      self._curr_obs.unsqueeze(0).to(self.device),
      action.unsqueeze(0).to(self.device))
    obs, reward, gamma = [x[0].cpu() for x in trans_out]

    self._curr_obs = obs

    # TODO: Fix this gamma stuff
    if gamma > 0.5:
      done = False
    else:
      done = True

    return obs, reward.item(), done, {}

class ObsEncoderWrapper(gym.ObservationWrapper):
  def __init__(self, env, encoder):
    super().__init__(env)
    self.encoder = encoder
    self.device = next(self.encoder.parameters()).device

    test_input = np.ones(env.observation_space.shape)
    test_input = preprocess_obs([test_input])
    with torch.no_grad():
      obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]
    self.observation_space = gym.spaces.Box(
      low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
    )

  def observation(self, obs):
    obs = preprocess_obs([obs])
    with torch.no_grad():
      obs = self.encoder.encode(obs.to(self.device))[0]
    return obs.cpu()

class DisallowActionWrapper(gym.ActionWrapper):
  def __init__(self, env, actions):
    super().__init__(env)

    self.act_mapping = []
    for i in range(self.action_space.n):
      if i not in actions:
        self.act_mapping.append(i)

    self.action_space = gym.spaces.Discrete(
      self.env.action_space.n - len(actions))

  def action(self, action):
    return self.act_mapping[action]

# Modified version of RGBImgObsWrapper from Minigrid
class MinigridRGBImgObsWrapper(ObservationWrapper):
  """
  Wrapper to use fully observable RGB image as observation,
  This can be used to have the agent to solve the gridworld in pixel space.
  """
  def __init__(self, env, tile_size=8, highlight=False):
    super().__init__(env)

    self.tile_size = tile_size
    self.highlight = highlight

    new_image_space = gym.spaces.Box(
      low=0,
      high=255,
      shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
      dtype='uint8',
    )

    self.observation_space = gym.spaces.Dict(
      {**self.observation_space.spaces, 'image': new_image_space})

  def observation(self, obs):
    env = self.unwrapped
    rgb_img = env.render(mode='rgb_array',
      highlight=self.highlight, tile_size=self.tile_size)
    return {**obs, 'image': rgb_img}

class CompactObsWrapper(gym.ObservationWrapper):
  """
  Fully observable gridworld using a compact grid encoding
  """

  def __init__(self, env, include_color=False):
    super().__init__(env)
    self.depth = 3 if include_color else 2

    obs_shape = (self.env.width, self.env.height, self.depth)
    low = np.zeros(obs_shape, dtype=np.float32)
    high = np.ones(obs_shape, dtype=np.float32)
    high[:, :, 0] *= 10 # OBJECT_TO_IDX
    high[:, :, self.depth-1] *= 3 # AGENT_DIR
    if include_color:
      high[:, :, 1] *= 5

    new_image_space = gym.spaces.Box(
      low=low, high=high, shape=obs_shape, dtype='float32')
    self.observation_space = gym.spaces.Dict(
      {**self.observation_space.spaces, "image": new_image_space}
    )

  def observation(self, obs):
    env = self.unwrapped
    full_grid = env.grid.encode()
    if self.depth == 3:
      full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX['agent'], 0, env.agent_dir]
      )
    elif self.depth == 2:
      full_grid = full_grid[:, :, [0, 2]]
      full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX['agent'], env.agent_dir]
      )

    return {**obs, "image": full_grid}
  
  def reverse_transform(obs):
    obs = obs.copy()
    if obs.shape[2] == 2:
      obs = np.stack([
        obs[:, :, 0],
        np.zeros((obs.shape[0], obs.shape[1]), dtype=obs.dtype),
        obs[:, :, 1],
      ], axis=2)

    agent_pos = np.stack(np.where(obs[:, :, 0] == OBJECT_TO_IDX['agent']))
    if len(agent_pos) == 0:
      agent_pos = None # [0, 0]
      agent_dir = None # 0
    else:
      agent_pos = np.stack(agent_pos).squeeze()
      agent_dir = obs[agent_pos[0], agent_pos[1], 2]
      obs[agent_pos[0], agent_pos[1]] = 0

    return obs, (agent_pos, agent_dir)
  
class UltraCompactObsWrapper(gym.ObservationWrapper):
  """
  Fully observable gridworld using a compact encoding
  """

  def __init__(self, env, key=False, door=False):
    super().__init__(env)
    self.is_key = key
    self.is_door = door

    self.key_pos = None
    self.door_pos = None

    self.codebook_size = max(env.width, env.height)

    # x, y, agent_dir (x4), key, door locked, door open
    obs_shape = (self.codebook_size, 3 + int(key) + 2*int(door))
    low = np.zeros(obs_shape, dtype=np.int64)
    high = np.ones(obs_shape, dtype=np.int64)

    new_image_space = gym.spaces.Box(
      low=low, high=high, shape=obs_shape, dtype='int64')
    self.observation_space = gym.spaces.Dict(
      {**self.observation_space.spaces, "image": new_image_space}
    )

  def reset(self, **kwargs):
    out = super().reset(**kwargs)

    # Get key and door pos
    if self.is_key or self.is_door:
      for i in range(self.env.width):
        for j in range(self.env.height):
          obj = self.env.grid.get(i, j)
          if obj is not None and obj.type == 'door':
            self.door_pos = (i, j)
          elif obj is not None and obj.type == 'key':
            self.key_pos = (i, j)
    
    return self.observation(out)

  def observation(self, obs):
    env = self.unwrapped

    x, y = env.agent_pos
    agent_dir = env.agent_dir

    compact_obs = [
      np.eye(self.codebook_size)[elem]
      for elem in (x, y, agent_dir)
    ]

    if self.is_key:
      if self.key_pos is None:
        compact_obs.append(np.eye(self.codebook_size)[0])
      else:
        obj = env.grid.get(*self.key_pos)
        if obj is not None and obj.type == 'key':
          compact_obs.append(np.eye(self.codebook_size)[1])
        else:
          compact_obs.append(np.eye(self.codebook_size)[0])

    # Check locked or not, open or closed
    if self.is_door:
      if self.door_pos is None:
        compact_obs.extend([
          np.eye(self.codebook_size)[0],
          np.eye(self.codebook_size)[0]
        ])
      else:
        obj = env.grid.get(*self.door_pos)
        if obj is not None and obj.type == 'door':
          if obj.is_locked:
            compact_obs.append(np.eye(self.codebook_size)[1])
          else:
            compact_obs.append(np.eye(self.codebook_size)[0])

          if obj.is_open:
            compact_obs.append(np.eye(self.codebook_size)[1])
          else:
            compact_obs.append(np.eye(self.codebook_size)[0])
        else:
            compact_obs.extend([
              np.eye(self.codebook_size)[0],
              np.eye(self.codebook_size)[0]
            ])

    compact_obs = np.array(compact_obs, dtype=np.int64).T

    return {**obs, "image": compact_obs}
  

class FreezeOnDoneWrapper(Wrapper):
  def __init__(self, env, max_count=-1):
    """Repeat last observation on done, until max_count is reached."""
    super().__init__(env)
    self.final_obs = None
    self.final_info = None
    self.max_count = max_count
    self.count = 0

  def reset(self, **kwargs):
    self.final_obs = None
    self.final_info = None
    self.count = 0
    return super().reset(**kwargs)

  def step(self, action):
    if self.final_obs is None:
      obs, reward, done, info = self.env.step(action)
      if done:
        self.final_obs = obs
        self.final_info = info
        self.count += 1
      return obs, reward, False, info
    
    done = self.max_count > 0 and self.count >= self.max_count
    return self.final_obs, 0, done, self.final_info

class RandomReseedWrapper(Wrapper):
  """
  Used to give an use different seeds for
  each n episodes.
  """

  def __init__(self, env, n_repeat=1):
    self._episode_idx = 0
    self.n_repeat = n_repeat
    self._seed = None
    super().__init__(env)

  def reset(self, **kwargs):
    if self._episode_idx % self.n_repeat == 0:
      self._seed = np.random.randint(0, int(1e9))
    self._episode_idx += 1
    return self.env.reset(seed=self._seed, **kwargs)
  

class MujocoObsWrapper(ObservationWrapper):
  SCALE_CONST = 1
  
  def __init__(self, env):
    super().__init__(env)

  def observation(self, observation):
    return observation / MujocoObsWrapper.SCALE_CONST
  
  def reverse_transform(observation):
    return observation * MujocoObsWrapper.SCALE_CONST


ENV_ALIASES = {
  'coinrun': 'procgen:procgen-coinrun-v0',
  'starpilot': 'procgen:procgen-starpilot-v0',
  'minigrid': 'MiniGrid-MultiRoom-N2-S4-v0',
  'minigrid-2-rooms': 'MiniGrid-MultiRoom-N2-S4-v0',
  'minigrid-4-rooms': 'MiniGrid-MultiRoom-N4-S5-v0',
  'minigrid-empty': 'MiniGrid-Empty-6x6-v0',
  'breakout': 'BreakoutNoFrameskip-v4',
  'crazyclimber': 'CrazyClimberNoFrameskip-v4',
  'mspacman': 'MsPacmanNoFrameskip-v4'
}

MINIGRID_KEY_ENVS = [
  'MiniGrid-KeyCorridorS3R1-v0', 'MiniGrid-KeyCorridorS3R2-v0',
  'MiniGrid-KeyCorridorS3R3-v0', 'MiniGrid-KeyCorridorS4R3-v0',
  'MiniGrid-KeyCorridorS5R3-v0', 'MiniGrid-KeyCorridorS6R3-v0'
]

MINIGRID_CROSSING_ENVS = [
  'MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-SimpleCrossingS9N2-v0',
  'MiniGrid-SimpleCrossingS9N3-v0', 'MiniGrid-SimpleCrossingS11N5-v0'
]

MINIGRID_COMPACT_ENVS = [
  'minigrid-crossing-stochastic-compactobs',
  'minigrid-door-key-stochastic-compactobs'
]

MINIGRID_UCOMPACT_ENVS = [
  'minigrid-crossing-stochastic-ucompact',
  'minigrid-door-key-stochastic-ucompact',
  'minigrid-crossing-stochastic-ucompact-flat',
  'minigrid-door-key-stochastic-ucompact-flat'
]

MUJOCO_ENVS = [
  'Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Humanoid-v4',
  'Reacher-v4', 'Walker2d-v4'
]

MUJOCO_VISUAL_ENVS = [
  x + '-Visual' for x in MUJOCO_ENVS
]

def check_env_name(env_name):
  return ENV_ALIASES.get(env_name, env_name)

def make_env(env_name, replay_buffer=None, buffer_lock=None, extra_info=None,
             monitor=False, max_steps=None):
  if env_name.lower() == 'minigrid-simple-stochastic':
    wrappers = [
      lambda env: MiniGridSimpleStochActionWrapper(env, n_acts=3),
      MinigridRGBImgObsWrapper,
      ImgObsWrapper,
      Custom2DWrapper
    ]
    env = gym.make('MiniGrid-Empty-6x6-v0')
    env.unwrapped.max_steps = max_steps or 10000
  elif env_name.lower().startswith('minigrid-key-stochastic'):
    # 6 Levels of difficulty
    if len(env_name) == len('minigrid-key-stochastic-X'):
      env_idx = int(env_name[-1]) - 1
    else:
      env_idx = 2
    tile_size = 8 if env_idx < 3 else 6

    wrappers = [
      MiniGridSimpleStochActionWrapper,
      lambda env: MinigridRGBImgObsWrapper(env, tile_size=tile_size),
      ImgObsWrapper,
      Custom2DWrapper,
      lambda env: ReseedWrapper(env, seeds=[41])
    ]  
    env = gym.make(MINIGRID_KEY_ENVS[env_idx])
    env.unwrapped.max_steps = max_steps or 10000
  elif env_name.lower().startswith('minigrid-crossing-stochastic'):
    if '-compactobs' in env_name.lower():
      compact = True
      env_name = env_name.replace('-compact_obs', '')
      img_wrapper = CompactObsWrapper
      scale_wrapper = lambda env: RescaleWrapper(
        env, np.array([[[0, 0]]]), np.array([[[10, 3]]]))
    elif '-ucompact' in env_name.lower():
      if '-flat' in env_name.lower():
        compact = True
        env_name = env_name.replace('-flat', '')
      else:
        compact = False
      env_name = env_name.replace('-ucompact', '')
      img_wrapper = UltraCompactObsWrapper
      scale_wrapper = lambda x: x
    else:
      compact = False
      img_wrapper = lambda env: MinigridRGBImgObsWrapper(env, tile_size=6)
      scale_wrapper = Custom2DWrapper

    if '-rand' in env_name.lower():
      env_name = env_name.replace('-rand', '')
      seeds = [np.random.randint(0, 1000000)]
    else:
      seeds = [41]

    wrappers = [
      lambda env: MiniGridSimpleStochActionWrapper(env, n_acts=3),
      img_wrapper,
      ImgObsWrapper,
      scale_wrapper,
      lambda env: ReseedWrapper(env, seeds=seeds)
    ]

    if compact:
      wrappers.append(FlattenObservation)

    # 4 Levels of difficulty
    if len(env_name) == len('minigrid-crossing-stochastic-X'):
      env_idx = int(env_name[-1]) - 1
    else:
      env_idx = 0     
    env = gym.make(MINIGRID_CROSSING_ENVS[env_idx])
    env.unwrapped.max_steps = max_steps or 10000
  elif env_name.lower().startswith('minigrid-door-key-stochastic'):
    if '-compactobs' in env_name.lower():
      compact = True
      env_name = env_name.replace('-compact_obs', '')
      img_wrapper = CompactObsWrapper
      scale_wrapper = lambda env: RescaleWrapper(
        env, np.array([[[0, 0]]]), np.array([[[10, 3]]]))
    elif '-ucompact' in env_name.lower():
      if '-flat' in env_name.lower():
        compact = True
        env_name = env_name.replace('-flat', '')
      else:
        compact = False
      env_name = env_name.replace('-ucompact', '')
      img_wrapper = lambda env: UltraCompactObsWrapper(
        env, key=True, door=True)
      scale_wrapper = lambda x: x
    else:
      compact = False
      img_wrapper = MinigridRGBImgObsWrapper
      scale_wrapper = Custom2DWrapper

    if '-rand' in env_name.lower():
      env_name = env_name.replace('-rand', '')
      seeds = [np.random.randint(0, 1000000)]
    else:
      seeds = [41]

    wrappers = [
      lambda env: DisallowActionWrapper(env, [4]),
      MiniGridSimpleStochActionWrapper,
      img_wrapper,
      ImgObsWrapper,
      scale_wrapper,
      lambda env: ReseedWrapper(env, seeds=seeds)
    ]
    
    if compact:
      wrappers.append(FlattenObservation)

    env = gym.make('MiniGrid-DoorKey-8x8-v0')
    env.unwrapped.max_steps = max_steps or 10000
  elif 'minigrid' in env_name.lower():
    scale_wrapper = Custom2DWrapper
    if '-fullobs' in env_name:
      env_name = env_name.replace('-fullobs', '')
      img_wrapper = MinigridRGBImgObsWrapper
    elif '-compactobs' in env_name:
      env_name = env_name.replace('-compactobs', '')
      img_wrapper = CompactObsWrapper
      scale_wrapper = lambda env: RescaleWrapper(
        env, np.array([[[0, 0]]]), np.array([[[10, 3]]]))
      # Axis 0: [0, 10], 1: 0, 2: [0, 3]
    else:
      img_wrapper = RGBImgPartialObsWrapper
    wrappers = [
      img_wrapper,
      ImgObsWrapper,
      scale_wrapper
    ]

    if '-seedrepeat-' in env_name:
      start_idx = env_name.find('-seedrepeat-')
      seed_start_idx = start_idx + len('-seedrepeat-')
      end_idx = env_name.find('-', seed_start_idx)
      if end_idx == -1:
        end_idx = len(env_name)
      n_repeat = int(env_name[seed_start_idx:end_idx])
      
      wrappers.append(lambda env: RandomReseedWrapper(env, n_repeat=n_repeat))
      env_name = env_name[:start_idx] + env_name[end_idx:]
    
    env = gym.make(env_name)
    env.unwrapped.max_steps = max_steps or 10000
  elif 'procgen' in env_name.lower():
    wrappers = [
      lambda env: Custom2DWrapper(env, frame_skip=4),
      lambda env: TransformObservation(env, torch.FloatTensor)
    ]
    env = gym.make(env_name)
  elif 'lunarlander' in env_name.lower():
    if 'frameskip' in env_name.lower():
      frame_skip = 4
      fs_idx = env_name.lower().find('frameskip')
      env_name = env_name[:fs_idx] + env_name[fs_idx + len('frameskip'):]
    else:
      frame_skip = 1
    wrappers = [
      RenderWrapper,
      lambda env: Custom2DWrapper(
        env, frame_skip=frame_skip, rescale_size=84, grayscale_obs=True),
      lambda env: FrameStack(env, num_stack=4),
      lambda env: TransformObservation(env, torch.FloatTensor),
      # Get rid of the extra grayscale dimension
      SqueezeDimWrapper
    ]
    env = gym.make(env_name)
  elif 'crafter' in env_name.lower():
    import crafter
    wrappers = [
      Custom2DWrapper,
      # lambda env: TransformObservation(env, torch.FloatTensor),
    ]
    env = gym.make(env_name)
  elif env_name in MUJOCO_ENVS:
    os.environ['MUJOCO_GL'] = 'osmesa'
    wrappers = [
      # MujocoObsWrapper,
      lambda env: TransformObservation(env, torch.FloatTensor)
    ]
    env = gym.make(env_name, new_step_api=False, width=120, height=120)
  elif env_name in MUJOCO_VISUAL_ENVS:
    os.environ['MUJOCO_GL'] = 'osmesa'
    wrappers = [
      RenderWrapper,
      lambda env: Custom2DWrapper(env, rescale_size=60, grayscale_obs=True),
      lambda env: FrameStack(env, num_stack=4),
      lambda env: TransformObservation(env, torch.FloatTensor),
      SqueezeDimWrapper
    ]
    env = gym.make(env_name[:-7], new_step_api=False, width=60, height=60)
  else:
    wrappers = [
      lambda env: AtariPreprocessing(env, scale_obs=True),
      lambda env: FrameStack(env, num_stack=4),
      lambda env: TransformObservation(env, torch.FloatTensor)
    ]
    env = gym.make(env_name)

  # Clip the reward
  env = TransformReward(env, lambda r: np.clip(r, -1, 1))
  if monitor:
    env = Monitor(env)

  if replay_buffer is not None:
    recorder_wrapper = lambda env: TrajectoryRecorderWrapper(
      env, replay_buffer, buffer_lock, extra_info)
    wrappers.append(recorder_wrapper)
    
  for wrapper in wrappers:
    env = wrapper(env)
  return env