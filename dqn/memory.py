"""Memory that stores episodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np
import tensorflow as tf

# Replay Buffer
from dqn.preprocess import atari_preprocess

episodes = 10000
capacity = 500000
frame_size = 4
replay_memory_init_size = 50000

transition = collections.namedtuple('transition', 'observ, reward, terminal, next_observ, action, advantage')
transition.__new__.__defaults__ = (None,) * len(transition._fields)


class ReplayBuffer(object):
    
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = collections.deque()
    
    def __len__(self):
        """Length of memory current size
        """
        return len(self._memory)
    
    def __getattr__(self, name):
        return getattr(self._memory, name)
    
    def append(self, _transition: transition):
        """
        Args:
            observ: Observation
            reward: Reward
            terminal: done
            next_observ: Next observation
            action: Action
            advantage: Adavantage Function
        """
        if len(_transition) != len(transition._fields):
            raise ValueError('transition given is not correct.')
        t = transition(*_transition)
        self._memory.append(t)
        if len(self._memory) > self._capacity:
            self._memory.popleft()
    
    def sample(self, batch_size):
        if len(self) < batch_size:
            raise ValueError('memory length is less than size.')
        choice = np.random.choice(len(self), batch_size)
        sample_ = np.asarray(self._memory)[choice]
        return sample_


def sample_action(observ, env):
    return env.action_space.sample()


# Collect transition and store in the replay buffer.
def initialize_memory(sess: tf.Session, env: gym.Env, batch_size=32):
    print('Initialize replay buffer memory...')
    replay_buffer = ReplayBuffer(capacity)
    
    observ = env.reset()
    observ = atari_preprocess(observ, sess)
    observ = np.stack([observ] * frame_size, axis=2)
    for t in itertools.count():
        print('\rMemory size {}'.format(len(replay_buffer)), end='', flush=True)
        if t >= replay_memory_init_size:
            break
        action = sample_action(observ, env)
        next_observ, reward, terminal, _ = env.step(action)
        next_observ = atari_preprocess(next_observ, sess)
        next_observ = np.concatenate(
            [observ[..., 1:], np.expand_dims(next_observ, axis=2)], axis=2)
        replay_buffer.append(
            transition(observ, reward, terminal, next_observ, action))
        if terminal:
            observ = env.reset()
            observ = atari_preprocess(observ, sess)
            observ = np.stack([observ] * frame_size, axis=2)
        else:
            observ = next_observ
    print('\rFinished initialize memory...')
    return replay_buffer


def main(_):
    init = tf.global_variables_initializer()
    env = gym.make('SpaceInvaders-v0')
    with tf.Session() as sess:
        init.run()
        buffer = initialize_memory(sess, env)
        print(len(buffer))


if __name__ == '__main__':
    tf.app.run()