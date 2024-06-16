import numpy as np
import torch


class ReplayBuffer():
  def __init__(self, capacity):
    self.capacity = capacity
    self.data = []
      
  def add_step(self, *step_data):
    self.data.append(step_data)
    if len(self.data) > self.capacity:
      self.data = self.data[-self.capacity:]
          
  def sample(self, n):
    n = min(n, len(self.data))
    indices = np.random.choice(range(len(self.data)), n, replace=False)
    samples = np.asarray(self.data)[indices]
    
    return_data = []
    n_elements = len(samples[0])
    for i in range(n_elements):
      return_data.append(
        torch.from_numpy(np.stack(samples[:, i])))
    
    return return_data

class SizedReplayBuffer():
  def __init__(self, capacity):
    self.capacity = capacity
    self.data = None
    self.n_elements = None
    self.curr_idx = 0 # Loops when full
    self.length = 0
      
  def create_buffers(self, *step_data):
    self.n_elements = len(step_data)
    self.data = []
    for i in range(self.n_elements):
      self.data.append(
        torch.zeros(self.capacity, *step_data[i].shape, dtype=step_data[i].dtype))

  def add_step(self, *step_data):
    if self.data is None:
      self.create_buffers(*step_data)
    
    for i in range(self.n_elements):
      self.data[i][self.curr_idx] = step_data[i]
    self.curr_idx = (self.curr_idx + 1) % self.capacity
    self.length = min(self.length + 1, self.capacity)
          
  def sample(self, n):
    n = min(n, self.length)
    indices = np.random.choice(range(self.length), n, replace=False)
    samples = []
    for i in range(self.n_elements):
      samples.append(self.data[i][indices])
    
    return samples