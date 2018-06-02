"""Construct Q-network and target network"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
from agents import tools


# Q-Network
def feed_forward_value(config,
                       observ_space: gym.Env.observation_space,
                       action_space: gym.Env.action_space,
                       action=None,
                       processed_observ_shape=(84, 84, 4)):
    """Independent feed forward networks for policy and value.
    
    The policy network outputs the mean action and the log standard deviation
    is learned as independent parameter vector.
    Args:
      config: Configuration object.
      action_space: Action space of the environment.
      observs: Sequences of observations.
      preprocess_fn: Pre-process the observation with TensorFlowed.
    Raises:
      ValueError: Unexpected action space.
    Returns:
      Attribute dictionary containing the policy, value, and unused state.
    """
    
    if not isinstance(observ_space, gym.spaces.Box):
        raise ValueError('Unexpected observation space.')
    if not isinstance(action_space, gym.spaces.Discrete):
        raise ValueError('Unexpected action space.')
    action_size = action_space.n
    observ = tf.placeholder(tf.float32, [None, *processed_observ_shape])
    if not isinstance(action, tf.Tensor):
        action = tf.placeholder(tf.int32, [None])  # [2, 1, 1, 0] if batch_size is 4
        feed = lambda observ_, action_: {observ: observ_, action: action_}
    else:
        print('action {}'.format(action))
        feed = lambda observ_: {observ: observ_}
    
    with tf.variable_scope('value'):
        x = observ
        for size in config.conv_value_layers:
            x = tf.layers.conv2d(x, filters=size[0], kernel_size=size[1], strides=size[2], activation=tf.nn.relu)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 512, tf.nn.relu)
        value = tf.layers.dense(x, action_size)
        value = tf.check_numerics(value, value.name)
        
        # Look at experimental/gather.py
        gather_indices = tf.range(config.batch_size) * action_size + action
        picked_value = tf.gather(value, gather_indices)
    
    with tf.variable_scope('policy'):
        picked_action = tf.argmax(value, axis=1)
        picked_action = tf.cast(picked_action, tf.int32)
    
    return tools.AttrDict(
        value=value, picked_value=picked_value, picked_action=picked_action, feed=feed)


def value_loss(network, target):
    pass