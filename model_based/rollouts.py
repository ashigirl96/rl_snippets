"""Rollout return one trajectory"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np
import tensorflow.contrib.distributions as tfd

transition = collections.namedtuple(
  'transition', 'observ, action, next_observ, reward, terminal')


class Experiment(object):
  _noise = tfd.Normal(0., 0.001)
  
  def __init__(self, env: gym.Env, use_monitor=False):
    """Experiment
    
    Args:
        env: OpenAI Gym.
        use_monitor:
    """
    self._env = env
    self._use_monitor = use_monitor
    self.episode_n = 0.
  
  def rollout(self, policy, random_trajectory=False):
    """Return Normalized trajectry.
    
    policy: Policy that returns probability.
    
    Returns:
        One trajectory.
    """
    if not isinstance(self._env.action_space, gym.spaces.Box):
      raise ValueError('This rollout is called only continuous ones.')
    if len(self._env.action_space.shape) > 1:
      raise NotImplementedError('Multi action cannot impemented.')
    
    if not random_trajectory:
      self.episode_n += 1
    
    observ = self._env.reset()
    if random_trajectory:
      observ += self._noise.sample(sample_shape=observ.shape)
    trajectory = []
    
    for t in itertools.count():
      action_prob = policy.predict(observ)
      action = action_prob.sample()
      assert action.shape == self._env.action_space.shape
      next_observ, reward, done, _ = self._env.step(action)
      trajectory.append(transition(observ, action, next_observ, reward, done))
      if done:
        break
    
    # Normalize observations.
    observs, actions, next_observs, _, _ = map(np.asarray, zip(*trajectory))
    normalize_observ = np.stack([observs, next_observs], axis=0).mean()
    normalize_action = actions.mean()
    for i, t in enumerate(trajectory):
      t = t._replace(observ=(t.observ / normalize_observ),
                     action=(t.action / normalize_action),
                     next_observ=(t.next_observ / normalize_observ),
                     )
      trajectory[i] = t
    
    return trajectory