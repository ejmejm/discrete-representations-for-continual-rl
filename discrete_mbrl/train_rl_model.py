import os
import sys
import time
import warnings
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from shared.models import *
from shared.trainers import *
from data_helpers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from utils import obs_to_img
from gym.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


SB3_DIR = 'mbrl_runs/'
N_EXAMPLE_ROLLOUTS = 4
N_EVAL_EPISODES = 20
EVAL_INTERVAL = 10
EVAL_UNROLL_STEPS = 20


def traj_to_imgs(obs_buffer, episode_starts, encoder,
                 n_imgs=N_EXAMPLE_ROLLOUTS, wandb_format=True):
    if isinstance(obs_buffer, np.ndarray):
        obs_buffer = torch.from_numpy(obs_buffer)
    idx = 0
    traj_imgs = []
    device = next(encoder.parameters()).device
    for _ in range(n_imgs):
        if idx >= len(obs_buffer):
            break
        traj_obs = [obs_buffer[idx]]

        for _ in range(EVAL_UNROLL_STEPS - 1):
            idx += 1
            if idx >= len(obs_buffer) or episode_starts[idx]:
                break
            traj_obs.append(obs_buffer[idx])

        traj_obs = torch.stack(traj_obs)
        # Shape: (EVAL_UNROLL_STEPS, channels, height, width)
        decoded_obs = encoder.decode(traj_obs.to(device))
        img = obs_to_img(decoded_obs, cat=True)
        if wandb_format:
            img = wandb.Image(img)
        traj_imgs.append(img)
    return traj_imgs

def rl_eval(rl_model, env, encoder, n_eval_episodes=10, log=True):
    obs_buffer = [[] for _ in range(env.num_envs)]
    episode_starts = [[] for _ in range(env.num_envs)]

    last_episode_starts = [True] * env.num_envs
    def callback(locals, _):
        nonlocal last_episode_starts
        curr_obs = locals['observations']
        dones = locals['dones']
        for i, o in enumerate(curr_obs):
            obs_buffer[i].append(o)
            episode_starts[i].append(last_episode_starts[i])
            last_episode_starts[i] = dones[i]

    ep_rewards, ep_lengths = evaluate_policy(rl_model, env, callback=callback,
        n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
    mean_reward = np.mean(ep_rewards)
    reward_std = np.std(ep_rewards)
    mean_len = np.mean(ep_lengths)

    if not log:
        return mean_reward, reward_std, mean_len

    # Shape: (n_envs, n_steps, obs_dim) -> (n_steps, obs_dim)
    obs_buffer = np.concatenate([
        np.stack(obs) for obs in obs_buffer], axis=0)
    # Shape: (n_envs, n_steps) -> (n_steps)
    episode_starts = np.concatenate([
        np.stack(starts) for starts in episode_starts], axis=0)

    traj_imgs = traj_to_imgs(obs_buffer, episode_starts, encoder)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wandb.log({
            'rl/eval/ep_reward_mean': np.mean(ep_rewards),
            'rl/eval/ep_reward_std': np.std(ep_rewards),
            'rl/eval/ep_mean_len': np.mean(ep_lengths),
            'rl/eval/trajectory': traj_imgs,
        })

    return mean_reward, reward_std, mean_len

class WandbCallback(BaseCallback):
    def __init__(self, encoder, eval_env=None, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.encoder = encoder
        self.eval_env = eval_env
        self.device = next(encoder.parameters()).device
        self.train_step = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ### Log RL Data ###

        learner = self.locals['self']
        ep_info_buffer = learner.ep_info_buffer
        fps = int((learner.num_timesteps - learner._num_timesteps_at_start) \
                   / (time.time() - learner.start_time))

        ### Log Sample Trajectories ###

        obs_buffer = self.locals['rollout_buffer'].observations
        # Shape: (self.buffer_size, self.n_envs) + self.obs_shape
        #     -> (self.n_envs * self.buffer_size, self.obs_shape)
        obs_buffer = obs_buffer.swapaxes(0, 1).reshape(obs_buffer.shape[0] * obs_buffer.shape[1], -1)

        # Shape: (self.buffer_size, self.n_envs)
        episode_starts = self.locals['rollout_buffer'].episode_starts.astype(bool)
        traj_imgs = traj_to_imgs(obs_buffer, episode_starts, self.encoder)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wandb.log({
                'rl/train/ep_reward_mean': np.mean([ep_info['r'] for ep_info in ep_info_buffer]),
                'rl/train/ep_mean_len': np.mean([ep_info['l'] for ep_info in ep_info_buffer]),
                'rl/train/fps': fps,
                'rl/train/step': learner.num_timesteps,
                'rl/train/imagined_trajectory': traj_imgs
            })

        # Log RL Eval Data
        if self.eval_env is not None and EVAL_INTERVAL > 0 and self.train_step % EVAL_INTERVAL == 0:
            model = self.locals['self']
            print('Running RL Eval...')
            mean_reward, std_reward, mean_len = rl_eval(
                model, self.eval_env, self.encoder, n_eval_episodes=N_EVAL_EPISODES, log=True)
            print('Episode reward mean: {:.3f} | reward std: {:.3f} | mean length: {:.2f}'.format(
                mean_reward, std_reward, mean_len))

        self.train_step += 1


def train_rl_model(args, encoder_model=None, trans_model=None):
    if args.wandb:
        global wandb
        import wandb
        
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n
    sample_obs = env.reset()
    sample_obs = preprocess_obs([sample_obs])
    
    # Load the encoder
    if encoder_model is None:
        encoder_model = construct_ae_model(
            sample_obs.shape[1:], args)[0]
    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()
    print(f'Loaded encoder')

    # Load the transition model
    if trans_model is None:
        trans_model = construct_trans_model(encoder_model, args, act_dim)[0]
    trans_model = trans_model.to(args.device)
    freeze_model(trans_model)
    trans_model.eval()
    print(f'Loaded transition model')

    if args.wandb:
        wandb.config.update(args, allow_val_change=True)

    world_model = PredictedModelWrapper(env, encoder_model, trans_model)
    if args.rl_unroll_steps > 0:
        world_model = TimeLimit(world_model, args.rl_unroll_steps)
    world_model = Monitor(world_model)

    model = PPO('MlpPolicy', world_model, verbose=1, device=args.device)

    # Uses real observations
    eval_env = DummyVecEnv([lambda: Monitor(ObsEncoderWrapper(env, encoder_model))])
    wandb_callback = WandbCallback(encoder_model, eval_env) if args.wandb else None
    
    try:
        model.learn(args.rl_train_steps, callback=wandb_callback)
    except KeyboardInterrupt:
        print('Stopping training')

    mean_reward, std_reward, mean_len = rl_eval(
        model, eval_env, encoder_model, n_eval_episodes=N_EVAL_EPISODES, log=args.wandb)
    print('Episode reward mean: {:.3f} | reward std: {:.3f} | mean length: {:.2f}'.format(
        mean_reward, std_reward, mean_len))

    return model

if __name__ == '__main__':
    # Parse args
    args = get_args()

    # Setup wandb
    if args.wandb:
        import wandb
        wandb.init(project='discrete-model-only-rl', config=args, tags=args.tags,
            settings=wandb.Settings(start_method='thread'), allow_val_change=True)
        args = wandb.config

    # Train and test the model
    model = train_rl_model(args)

    # Model saving and loading currently not implemented