"""Test replay buffer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
from agents import tools

from dqn.memory import ReplayBuffer, initialize_memory


class ReplayBufferTest(tf.test.TestCase):
    def test_capacity(self):
        replay_buffer = ReplayBuffer(2)
        transition1 = (1, 2, 3, 4, 5, 6)
        replay_buffer.append(transition1)
        transition2 = (11, 2, 3, 4, 5, 6)
        replay_buffer.append(transition2)
        transition3 = (21, 2, 3, 4, 5, 6)
        replay_buffer.append(transition3)
        assert len(replay_buffer) == 2, len(replay_buffer)
    
    def test_batch_size_greater_length(self):
        replay_buffer = ReplayBuffer(10)
        transition1 = (1, 2, 3, 4, 5, 6)
        replay_buffer.append(transition1)
        transition2 = (11, 2, 3, 4, 5, 6)
        replay_buffer.append(transition2)
        transition3 = (21, 2, 3, 4, 5, 6)
        replay_buffer.append(transition3)
        with self.assertRaises(ValueError):
            replay_buffer.sample(4)


def default_config():
    # For Training
    episodes = 10000
    batch_size = 32
    # Replay Buffer
    capacity = 500000
    frame_size = 4
    replay_memory_init_size = 35
    return locals()


def test_samples():
    env = gym.make('SpaceInvaders-v0')
    config = tools.AttrDict(default_config())
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        buffer = initialize_memory(sess, env, config)
        batch_transition = buffer.sample(config.batch_size)
        
        assert batch_transition.action.shape == (config.batch_size,)
        assert batch_transition.observ.shape == (32, 84, 84, 4)


def main(_):
    test_samples()

if __name__ == '__main__':
    # tf.test.main()
    tf.app.run()