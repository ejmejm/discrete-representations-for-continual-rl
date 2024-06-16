import os
import sys
sys.path.append('../')

import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch import nn

from env_helpers import make_env


MODEL_SAVE_FORMAT = os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  'models/{}/ppo_{}.sb3')


class RandomPolicy(nn.Module):
  def __init__(self, env_name):
    super().__init__()
    env = make_env(env_name)
    self.action_space = env.action_space

  def act(self, obs):
    acts = [self.action_space.sample() for _ in range(len(obs))]
    return torch.tensor(acts).long().to(obs.device)

STATIC_POLICY_REGISTRY = {
  'random': RandomPolicy,
}


def load_policy(env_name, goal_type):
  if goal_type in STATIC_POLICY_REGISTRY:
    return STATIC_POLICY_REGISTRY[goal_type](env_name)

  model_path = MODEL_SAVE_FORMAT.format(env_name, goal_type)
  print(model_path)
  if not os.path.exists(model_path):
    raise ValueError(f'No model found for {env_name} with goal type "{goal_type}"!')
  policy = ActorCriticPolicy.load(model_path)

  def act(obs):
    fobs = format_sb3_obs(obs)
    acts = policy.predict(fobs, deterministic=False)[0]
    return torch.from_numpy(acts).long().to(obs.device)
  policy.act = act

  return policy

# SB3 only takes in a certain obs format (for `CNNPolicy`)
# This function will conver to that format
def format_sb3_obs(obs):
  if isinstance(obs, torch.Tensor):
    obs = obs.cpu().numpy()
  elif not isinstance(obs, np.ndarray):
    obs = np.array(obs)
  return (obs * 255).astype(np.uint8)