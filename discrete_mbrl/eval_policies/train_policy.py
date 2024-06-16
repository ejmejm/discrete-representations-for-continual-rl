# python train_policy.py --env_name X --goal_type Y
# Get the env
# Create a list of different reward wrappers
# Apply the reward wrappers to the env based on the goal type
# Create the policy with SB3
# Train the policy, and save it to the models directory

import argparse
import os
import sys
import zipfile
sys.path.append('../')

from gym.wrappers import RecordEpisodeStatistics
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


from env_helpers import make_env
from policies import MODEL_SAVE_FORMAT
from wrappers import SB3ObsWrapper, apply_goal_wrapper


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '-e', type=str, required=True)
  parser.add_argument('--goal_type', '-g', type=str, required=True)
  parser.add_argument('--n_envs', '-n', type=int, default=8)
  parser.add_argument('--train_steps', '-s', type=int, default=int(3e5))
  parser.add_argument('--eval_freq', type=int, default=30_000)
  parser.add_argument('--eval_episodes', type=int, default=20)
  parser.add_argument('--policy_model', type=str, default='CnnPolicy')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  
  # Create env
  env = make_env(args.env_name, max_steps=args.env_max_steps)
  env = apply_goal_wrapper(env, args.goal_type)
  env = SB3ObsWrapper(env)
  
  env = RecordEpisodeStatistics(env)
  env = SubprocVecEnv([lambda: env for _ in range(args.n_envs)])

  # Create callback to save highest reward model
  eval_callback = EvalCallback(
    env, n_eval_episodes=args.eval_episodes, eval_freq=args.eval_freq,
    best_model_save_path='./logs/', log_path='./logs/',
    deterministic=True)

  # Create and train policy
  model = PPO(args.policy_model, env, verbose=1, tensorboard_log='./logs/',)
  model.learn(total_timesteps=args.train_steps, callback=eval_callback)

  # Load the best policy and eval stats
  model = PPO.load('./logs/best_model.zip')
  eval_stats = np.load('./logs/evaluations.npz')

  # Print best model stats
  rewards = eval_stats['results']
  ep_lens = eval_stats['ep_lengths']
  mean_rewards = rewards.mean(axis=1)
  best_idx = np.argmax(mean_rewards)
  print('Best Model Stats:')
  print(f'Timestep: {eval_stats["timesteps"][best_idx]}')
  print('Reward mean: {:.2f} | std: {:.2f} | min: {:.2f} | max: {:.2f}'.format(
    mean_rewards[best_idx], rewards[best_idx].std(),
    rewards[best_idx].min(), rewards[best_idx].max()))
  print('Episode length mean: {:.2f} | std: {:.2f} | min: {:.2f} | max: {:.2f}'.format(
    ep_lens[best_idx].mean(), ep_lens[best_idx].std(),
    ep_lens[best_idx].min(), ep_lens[best_idx].max()))

  # Save policy
  save_path = MODEL_SAVE_FORMAT.format(args.env_name, args.goal_type)
  save_dir = os.path.dirname(save_path)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  policy = model.policy
  policy.save(save_path)

  # Delete the logged data
  os.remove('./logs/best_model.zip')
  os.remove('./logs/evaluations.npz')