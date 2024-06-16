import copy
import cv2
from gym_minigrid.minigrid import Grid
import numpy as np
import torch

from env_helpers import (
  make_env,
  MUJOCO_ENVS,
  MINIGRID_COMPACT_ENVS,
  MINIGRID_UCOMPACT_ENVS,
  CompactObsWrapper
)


cached_env = None
MUJOCO_IMG_DIM = (120, 120)


def mujoco_state_to_img(
  state, env_name=None, transform=None, img_dim=None,
  x_pos=0, y_pos=0, **kwargs):

  global cached_env
  if cached_env is None:
    cached_env = make_env(env_name)
  
  img_dim = img_dim or MUJOCO_IMG_DIM

  transform = transform or (lambda x: x)
  if isinstance(state, torch.Tensor):
    state = transform(state.cpu()).numpy()
  else:
    state = transform(state)
  
  cached_env.reset()
  n_pos = cached_env.model.nq - 2
  n_vel = cached_env.model.nv
  pos = np.array([x_pos, y_pos] + state[:n_pos].tolist())
  vel = np.array(state[n_pos:n_pos+n_vel].tolist())
  
  cached_env.set_state(pos, vel)
  img = cached_env.render(
    'rgb_array', width=img_dim[1], height=img_dim[0])
  img = img.astype(np.float32) / 255
  if img.shape[:2] != img_dim:
    img = cv2.resize(img, img_dim)
  img = img.transpose(2, 0, 1)
  return img

def apply_obs_transforms(obs, env):
  """Apply all of the wrapper observation transforms
     from a wrapped environment."""
  transforms = [None]
  while getattr(env, 'env', None) is not None:
    # Check if env has observation function
    if hasattr(env, 'observation') and env.observation != transforms[-1]:
      transforms.append(env.observation)
    env = env.env
  transforms = transforms[1:]

  for transform in transforms[::-1]:
    obs = transform(obs)

  return obs

def minigrid_compact_state_to_img(state, env_name=None, transform=None, **kwargs):
  global cached_env
  if cached_env is None:
    cached_env = make_env(env_name.replace('-compactobs', ''))
  
  transform = transform or (lambda x: x)
  state = transform(state.cpu()).numpy()

  cached_env.reset()

  width = cached_env.grid.width
  height = cached_env.grid.height
  state = state.reshape(height, width, -1)
  n_channels = state.shape[-1]
  print('a')
  print(state.shape)

  state[:, :, 0] = state[:, :, 0].clip(0, 1)
  state[:, :, 0] *= 10
  if n_channels == 3: # Color channel doesn't always exist
    state[:, :, 1] = state[:, :, 1].clip(0, 1)
    state[:, :, 1] *= 5
  state[:, :, -1] = state[:, :, -1].clip(0, 1)
  state[:, :, -1] *= 3
  state = state.round().astype(np.uint8)
  print('b')
  print(state.shape, state[0])

  grid, (agent_pos, agent_dir) = CompactObsWrapper.reverse_transform(state)
  print(agent_pos, agent_dir)
  grid = Grid.decode(grid)[0]
  cached_env.unwrapped.grid = grid
  cached_env.unwrapped.agent_pos = agent_pos
  cached_env.unwrapped.agent_dir = agent_dir

  raw_obs = cached_env.gen_obs() # {'image': cached_env.gen_obs()}
  obs = apply_obs_transforms(raw_obs, cached_env)

  return obs

def minigrid_ucompact_state_to_img(state, env_name=None, transform=None, **kwargs):
  global cached_env
  if cached_env is None:
    cached_env = make_env(
      env_name
      .replace('-ucompact', '')
      .replace('-flat', '')
    )
  
  transform = transform or (lambda x: x)
  state = transform(state.cpu()).numpy()

  cached_env.reset()
  w = cached_env.width
  h = cached_env.height

  state = state.reshape(max(w, h), -1)

  x = min(np.argmax(state[:, 0]), w)
  y = min(np.argmax(state[:, 1]), h)

  cached_env.unwrapped.agent_pos = np.array([x, y], dtype=np.int32)
  cached_env.unwrapped.agent_dir = min(np.argmax(state[:, 2]), 3)

  # Check if door key env
  if len(state) == 3:
    # Get door and key positions
    for i in range(cached_env.env.width):
      for j in range(cached_env.env.height):
        obj = cached_env.unwrapped.grid.get(i, j)
        if obj is not None and obj.type == 'door':
          door = obj
        elif obj is not None and obj.type == 'key':
          key = obj
          key_pos = (i, j)
          
    # Get door and key states
    key_on_ground = bool(np.argmax(state[:, 3]))
    door_locked = bool(np.argmax(state[:, 4]))
    door_open = bool(np.argmax(state[:, 5]))
    
    # Set states in env
    if not door_locked:
      door.is_locked = False

    if door_open:
      door.is_open = True
    
    if not key_on_ground:
      cached_env.unwrapped.grid.set(*key_pos, None)
      cached_env.unwrapped.carrying = key

  raw_obs = cached_env.gen_obs()
  obs = apply_obs_transforms(raw_obs, cached_env)

  return obs

def default_state_to_img(state, **kwargs):
  """Input is a single state, output is (c x h x w) image."""
  if isinstance(state, torch.Tensor):
    state = state.cpu().numpy()
  return state


env_transform_map = {
  None: default_state_to_img,
  **{e: minigrid_compact_state_to_img for e in MINIGRID_COMPACT_ENVS},
  **{e: minigrid_ucompact_state_to_img for e in MINIGRID_UCOMPACT_ENVS},
  **{e: mujoco_state_to_img for e in MUJOCO_ENVS}
}


def get_img_transform(env_name):
  return env_transform_map.get(env_name, default_state_to_img)

def state_to_img(state, env_name=None, **kwargs):
  return get_img_transform(env_name)(state, env_name=env_name, **kwargs)

def states_to_imgs(states, env_name=None, **kwargs):
  transform = get_img_transform(env_name)
  imgs = [transform(s, env_name=env_name, **kwargs) for s in states]
  imgs = np.stack(imgs)
  return imgs