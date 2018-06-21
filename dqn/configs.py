"""Configurations for construct Deep Q-Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class default_config:
  conv_value_layers = [
    [32, 8, 4], [64, 4, 2], [64, 3, 1]]
  batch_size = 32
  gamma = 0.99
  grad_norm_clipping = 10
  stopping_crierion = None
  
  update_target_estimator_every = 10_000
  
  # Train episodes
  num_episodes = 10_000
  
  # Replay Buffer
  capacity = 1_000_000
  frame_size = 4
  frame_dim = 84
  replay_memory_init_size = 50_000
  # replay_memory_init_size = 1_00
  # eps-greedy policy coefficient
  epsilon_start = 1.0
  epsilon_end = 0.1
  epsilon_decay_steps = 500_000
  
  use_gpu = True
  use_dddqn = False
  
  # max step size
  max_total_step_size = 1_000_000