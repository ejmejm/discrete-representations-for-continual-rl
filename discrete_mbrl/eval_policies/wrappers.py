from collections import Counter
import math
import sys
sys.path.append('../')

import numpy as np
from gym import spaces, Wrapper, ObservationWrapper


class SB3ObsWrapper(ObservationWrapper):
  """ Converts observations to the SB3 required format. """
  def __init__(self, env):
    super().__init__(env)
    if (self.observation_space.high == 1).all():
      self.observation_space = spaces.Box(
        low=0, high=255,
        shape=self.observation_space.shape,
        dtype=np.uint8)
      self.observation = self._convert_obs
  
  def _convert_obs(self, obs):
    return (obs * 255).astype(np.uint8)

# Based off of https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/wrappers.py
class ExploreRight(Wrapper):
    """
    Changes the reward to give a reward of 1 / sqrt(count) for each step
    taken on the right half of the grid, and -0.1 for everything else.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = Counter()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        pos = tuple(env.agent_pos)

        if pos[0] >= env.grid.width // 2:
          self.counts[pos] += 1
          reward = 1 / math.sqrt(self.counts[pos])
        else:
          reward = -0.1

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


GOAL_REGISTRY = {
  'explore_right': ExploreRight,
  'goal': lambda x: x
}

def apply_goal_wrapper(env, goal_type):
  return GOAL_REGISTRY[goal_type](env)

# if __name__ == '__main__':
#   import gym
#   import gym_minigrid
#   from env_helpers import make_env
#   import matplotlib.pyplot as plt
#   # Test the state bonus
#   env = make_env('minigrid-crossing-stochastic')
#   env.reset()
#   for _ in range(100):
#     env.step(env.action_space.sample())
#     print(env.unwrapped.agent_pos)
#     print(env.unwrapped.grid)
#     env.render('human')
#     plt.show()
#     import time
#     time.sleep(1)
#     # 0, 0 indexed from the top left