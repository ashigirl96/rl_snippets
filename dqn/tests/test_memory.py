"""Test replay buffer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dqn.memory import ReplayBuffer


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


if __name__ == '__main__':
    tf.test.main()