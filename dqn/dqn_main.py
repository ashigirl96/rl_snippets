"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import gym
import tensorflow as tf
from agents import tools
import numpy as np

from dqn.configs import default_config
from dqn.memory import initialize_memory
from dqn.networks import ValueFunction
from dqn.preprocess import atari_preprocess


def make_session(num_cpu=8):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def initialize_variables(sess: tf.Session):
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
    ])


def main(_):
    env = gym.make('SpaceInvaders-v0')
    
    _config = tools.AttrDict(default_config())
    
    # Initialize networks.
    with tf.variable_scope('q_network'):
        q_network = ValueFunction(_config, env.observation_space, env.action_space)
    with tf.variable_scope('target'):
        target = ValueFunction(_config, env.observation_space, env.action_space, q_network)
    sess = make_session()
    initialize_variables(sess)
    # Initialize memory
    memory = initialize_memory(sess, env, _config)
    
    for episode in range(_config.episodes):
        observ = env.reset()
        observ = atari_preprocess(sess, observ)
        observ = np.stack([observ] * 4, axis=2)
        
        print(observ.shape)
        
        # for t in itertools.count():
        #     pass


if __name__ == '__main__':
    tf.app.run()