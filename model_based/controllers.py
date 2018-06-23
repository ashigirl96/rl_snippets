"""Controllers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from model_based.cost_function import trajectory_cost_fn
from model_based.dynamics import DynamicsNetwork


class Controller(object):
  def __init__(self, env: gym.Env):
    """Random action policy according environment.
    
    Args:
        env: OpenAI Gym.
    """
    self._env = env
  
  def get_action(self, sess, observ):
    pass


class MPCcontroller(object):
  """
  Controller built using the MPC method outlined in
  https://arxiv.org/abs/1708.02596
  """
  
  def __init__(self,
               env,
               dynamics: DynamicsNetwork,
               horizon=5,
               cost_fn=None,
               num_simulated_paths=10):
    self._env = env
    self._dynamics = dynamics
    self._horizon = horizon
    self._cost_fn = cost_fn
    self._num_simulated_paths = num_simulated_paths
  
  def get_action(self, sess, observ):
    # Note: be careful to batch your simulations through the model for
    # speed
    all_observs = []
    all_actions = []
    all_next_states = []
    
    observs = np.array([observ] * self._num_simulated_paths)
    for _ in range(self._horizon):
      actions = np.array([self._env.action_space.sample() for _ in range(self._num_simulated_paths)])
      next_observs = self._dynamics.predict(sess, observs, actions)
      
      all_observs.append(observs)
      all_actions.append(actions)
      all_next_states.append(next_observs)
      
      observs = next_observs
    
    costs = trajectory_cost_fn(all_observs, all_actions, all_next_states)
    return all_actions[0][np.argmin(costs)]


class RandomController(Controller):
  def __init__(self, env: gym.Env):
    super(RandomController, self).__init__(env)
  
  def get_action(self, unused_sess, unused_observ):
    return self._env.action_space.sample()