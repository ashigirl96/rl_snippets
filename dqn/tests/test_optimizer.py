"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.utils.atari_wrappers import wrap_deepmind

from dqn.configs import default_config
from dqn.networks import ValueFunction


def main(_):
    env = gym.make('SpaceInvaders-v0')
    # env = gym.make('Pong-v0')
    env = wrap_deepmind(env)
    
    experiment_dir = os.path.abspath("./experiments2/{}".format(env.spec.id))
    atari_actions = np.arange(env.action_space.n, dtype=np.int32)
    # _config = tools.AttrDict(default_config())
    _config = default_config()
    
    # Initialize networks.
    with tf.variable_scope('q_network'):
        q_network = ValueFunction(_config,
                                  env.observation_space,
                                  env.action_space,
                                  summaries_dir=experiment_dir)
    


if __name__ == '__main__':
    tf.app.run()