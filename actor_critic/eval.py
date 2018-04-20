"""Evaluate the policy trained."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from agents.tools import AttrDict

from reinforce.policy import Policy
from reinforce.rollout import video_evaluate_policy


def default_config():
    # Whether use bias on layer
    use_bias = True
    # OpenAI Gym environment name
    env_name = 'CartPole-v1'
    # Discount Factor (gamma)
    discount_factor = .995
    # Learning rate
    learning_rate = 0.1
    # Number of episodes
    num_episodes = 100
    # Activation function used in dense layer
    activation = tf.nn.relu
    # Epsilon-Greedy Policy
    eps = 0.1
    
    # Use GPUS
    use_gpu = True
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    
    return locals()


def main(_):
    config = AttrDict(default_config())
    policy = Policy(config)
    saver = tf.train.Saver()
    saver.restore(policy._sess, './checkpoints/reinforce_debug')
    
    frames = video_evaluate_policy(policy, config.env_name)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()