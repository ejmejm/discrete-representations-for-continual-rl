import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F


def ortho_init(ae, policy, critic):
  """
  Initialize the weights of the actor-critic and autoencoder
  models using orthogonal initialization
  """
  for m in ae.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
      torch.nn.init.zeros_(m.bias)

  for m in policy.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      # Check if last layer
      if m == policy[-1]:
        torch.nn.init.orthogonal_(m.weight, gain=0.01)
        torch.nn.init.zeros_(m.bias)
      else:
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(m.bias)

  for m in critic.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      # Check if last layer
      if m == critic[-1]:
        torch.nn.init.orthogonal_(m.weight, gain=1)
        torch.nn.init.zeros_(m.bias)
      else:
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(m.bias)


class PPOTrainer():
  def __init__(
      self, env, policy, critic, ae, optimizer,
      epsilon=1e-7, ppo_iters=20, ppo_clip=0.2, value_coef=0.5,
      minibatch_size=32, entropy_coef=0.003, gae_lambda=0,
      norm_advantages=False, max_grad_norm=0.5, e2e_loss=False
    ):
    self.n_acts = env.action_space.n
    self.policy = policy
    self.critic = critic
    self.ae = ae
    self.optimizer = optimizer
    self.device = next(self.policy.parameters()).device
    self.epsilon = epsilon
    self.ppo_iters = ppo_iters
    self.ppo_clip = ppo_clip
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.minibatch_size = minibatch_size
    self.gae_lambda = gae_lambda
    self.norm_advantages = norm_advantages
    self.max_grad_norm = max_grad_norm
    self.e2e_loss = e2e_loss
    self.policy_losses = []
    self.critic_losses = []
    self.step_idx = 1

  
  def calculate_gaes(self, rewards, values, next_values, gammas, decay=0.95):
    """
    Return the General Advantage Estimates from the 
    given reward and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    deltas = [rewards[i] + gammas[i] * next_values[i] - values[i] \
      for i in range(len(rewards))]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
      gaes.append(deltas[i] + decay * gammas[i] * gaes[-1])

    return torch.tensor(gaes[::-1])

  def train(self, batch_data):
    """
    Update the policy and critic models using the PPO algorithm

    Args:
      batch_data: A dict of tensors containing the following data:
        obs, states, act_log_probs, next_obs, rewards, acts, gammas
    """
    self.policy.train()

    # Bootstrap rewards if episode is not done
    if batch_data['gammas'][-1] > 0:
      with torch.no_grad():
        next_state = self.ae.encode(
          batch_data['next_obs'][-1:].to(self.device),
          return_one_hot=True)
        next_value = self.critic(next_state).squeeze()
      next_value = next_value.cpu()
      batch_data['rewards'][-1] += batch_data['gammas'][-1] * next_value
      batch_data['gammas'][-1] = 0

    # Calculate returns
    returns = torch.zeros(len(batch_data['rewards']))
    returns[-1] = batch_data['rewards'][-1]
    for i in reversed(range(len(batch_data['rewards']) - 1)):
      returns[i] = batch_data['rewards'][i] + \
        batch_data['gammas'][i] * returns[i+1]
      
    obs = batch_data['obs'].to(self.device)
    states = batch_data['states'].to(self.device)
    acts = batch_data['acts'].to(self.device)
    returns = returns.to(self.device)

    with torch.no_grad():
      values = self.critic(states).squeeze(1)
      logits = self.policy(states)
    probs = F.softmax(logits, dim=-1)
    old_act_probs = probs.gather(1, acts).squeeze(1)

    # Calculate advantages
    
    if self.gae_lambda == 0:
      with torch.no_grad():
        advantages = returns - values
    else:
      with torch.no_grad():
        last_state = self.ae.encode(
          batch_data['next_obs'][-1:].to(self.device),
          return_one_hot=True)
        last_value = self.critic(last_state).squeeze(dim=0)

      next_values = torch.cat([values[1:], last_value])
      advantages = self.calculate_gaes(
        batch_data['rewards'], values, next_values,
        batch_data['gammas'], decay=self.gae_lambda)
      advantages = advantages.to(self.device)

      returns = advantages + values

    train_data = {
      'obs': obs,
      'states': states,
      'acts': acts,
      'old_act_probs': old_act_probs,
      'old_values': values,
      'advantages': advantages,
      'returns': returns}

    policy_losses = []
    critic_losses = []
    entropy_losses = []

    for _ in range(self.ppo_iters):
      zipped_buffer = list(zip(*train_data.values()))
      np.random.shuffle(zipped_buffer)
      train_buffer = list(zip(*zipped_buffer))
      train_data = {k: torch.stack(v) for k, v in \
        zip(train_data.keys(), train_buffer)}

      # Break the data into mini-batches for the model updates
      for batch_idx in range(int(np.ceil(len(train_data['states']) / self.minibatch_size))):
        minibatch = {k: v[batch_idx * self.minibatch_size: \
          (batch_idx + 1) * self.minibatch_size] \
          for k, v in train_data.items()}
          
        # Calculate new action probabilities and values for the epoch
        if self.e2e_loss:
          minibatch['states'] = self.ae.encode(minibatch['obs'])
        new_values = self.critic(minibatch['states'])
        new_act_probs = F.softmax(self.policy(minibatch['states']), dim=-1)
        policy_entropy = Categorical(probs=new_act_probs).entropy()
        new_act_probs = new_act_probs.gather(1, minibatch['acts'])
        new_act_probs = new_act_probs.squeeze(1)
        new_values = new_values.squeeze(1)

        # Calulcate the value loss
        value_loss = F.mse_loss(new_values, minibatch['returns'])

        if self.norm_advantages:
          mb_advantages = (minibatch['advantages'] - minibatch['advantages'].mean()) \
            / (minibatch['advantages'].std() + 1e-8)
        else:
          mb_advantages = minibatch['advantages']

        # Calculate the policy loss
        # print(new_act_probs, minibatch['old_act_probs'])
        policy_ratio = new_act_probs / (minibatch['old_act_probs'] + self.epsilon)
        clipped_policy_ratio = torch.clamp(policy_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        policy_loss = torch.min(policy_ratio * mb_advantages,
          clipped_policy_ratio * mb_advantages)
        policy_loss = -policy_loss.mean()
        entropy_loss = -policy_entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss \
          + self.entropy_coef * entropy_loss

        policy_losses.append(policy_loss.item())
        critic_losses.append(value_loss.item())
        entropy_losses.append(entropy_loss.item())

        # Update the model
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    return {
      'policy_loss': np.mean(policy_losses),
      'critic_loss': np.mean(critic_losses),
      'entropy_loss': np.mean(entropy_losses)
    }