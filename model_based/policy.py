"""Model-free RL algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd


def _feed_forward_network(observ, action_size, init_output_factor=0.1, init_std=0.35):
  init_output_weights = tf.contrib.layers.variance_scaling_initializer(
    factor=init_output_factor)
  before_softplus_std_initializer = tf.constant_initializer(
    np.log(np.exp(init_std) - 1))
  
  x = tf.layers.dense(observ, 1000, tf.nn.relu)
  x = tf.layers.dense(x, 500, tf.nn.relu)
  x = tf.layers.dense(x, 100, tf.nn.relu)
  mean = tf.contrib.layers.fully_connected(
    x, action_size, tf.tanh,
    weights_initializer=init_output_weights)
  std = tf.nn.softplus(tf.get_variable(
    'before_softplus_std', mean.shape[2:], tf.float32,
    before_softplus_std_initializer))
  std = tf.tile(
    std[None, None],
    [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
  policy = tfd.MultivariateNormalDiag(mean, std)
  return policy


class PPO(object):
  
  def __init__(self, env: gym.Env,
               log_dir='./logdir'):
    if not isinstance(env.action_space, gym.spaces.Box):
      raise ValueError('This rollout is called only continuous ones.')
    if len(env.action_space.shape) > 1:
      raise NotImplementedError('Multi action cannot implemented.')
    if len(env.observation_space.shape) > 1:
      raise NotImplementedError('Multi obsrvation cannot implemented.')
    self._env = env
    self._log_dir = log_dir
    
    policy_summary = self._forward()
  
  def _forward(self):
    action_size = self._env.action_space.shape[0]
    self._observ_size = self._env.observation_space.shape[0]
    with tf.variable_scope('forward_policy_network'):
      self._observ = tf.placeholder(tf.float32, [None, self._observ_size])
      self._policy = _feed_forward_network(self._observ, action_size)
    
    policy_summary = tf.summary.merge([
      tf.summary.histogram('policy_samples', self._policy.sample()),
      tf.summary.histogram('policy_mode', self._policy.mode()),
    ])
    return policy_summary
  
  def _loss(self):
    pass


class RandomPolicy:
  
  def __init__(self, env: gym.Env):
    """Random action policy according environment.
    
    Args:
        env: OpenAI Gym.
    """
    self._env = env
  
  def predict(self, unused_observ):
    """Return random sample distribution."""
    return self._env.action_space


def main(_):
  pass


if __name__ == '__main__':
  tf.app.run()