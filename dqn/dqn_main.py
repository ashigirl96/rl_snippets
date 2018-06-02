"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf


# Replay Buffer


# replay_memory_init_size = 2


# Q-Network

# Target-Network

# optimizer = tf.train.AdamOptimizer(0.0001)
# train_op = optimizer.minimize(tf.Variable(0.1))


def main(_):
    init = tf.global_variables_initializer()
    env = gym.make('SpaceInvaders-v0')
    with tf.Session() as sess:
        init.run()
        buffer = initialize_memory(sess, env)
        print(len(buffer))


if __name__ == '__main__':
    tf.app.run()