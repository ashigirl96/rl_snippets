"""Memory that stores episodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np
import tensorflow as tf

from dqn.configs import default_config
from dqn import utility

# Replay Buffer
transition = collections.namedtuple('transition', 'observ, reward, terminal, next_observ, action, advantage')
transition.__new__.__defaults__ = (None,) * len(transition._fields)


class ReplayBuffer(object):
  
  def __init__(self, capacity, clip_reward=True):
    self._capacity = capacity
    self._clip_reward = clip_reward
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
  
  def sample(self, batch_size, unique=True):
    if len(self) < batch_size:
      raise ValueError('memory length is less than size.')
    choice = np.random.choice(len(self), batch_size, replace=unique)
    # 'transition', 'observ, reward, terminal, next_observ, action, advantage')
    observ, reward, terminal, next_observ, action, advantage = zip(*np.asarray(self._memory)[choice])
    if self._clip_reward:
      reward = np.sign(reward)
    batch_transition = transition(
      observ=np.asarray(observ),
      reward=reward,
      terminal=terminal,
      next_observ=np.asarray(next_observ),
      action=np.asarray(action),
      advantage=np.asarray(advantage),
    )
    return batch_transition


def sample_action(unused_observ, env):
  return env.action_space.sample()


# Collect transition and store in the replay buffer.
def initialize_memory(sess: tf.Session, env: gym.Env, config, q_network):
  capacity = config.capacity
  replay_memory_init_size = config.replay_memory_init_size
  
  print('Initialize replay buffer memory...')
  replay_buffer = ReplayBuffer(capacity)
  
  observ = env.reset()
  observ = observ
  for t in itertools.count():
    print('\rMemory size {}'.format(len(replay_buffer)), end='', flush=True)
    if t >= replay_memory_init_size:
      break
    action = q_network.best_action(sess, observ[None])[0]
    next_observ, reward, terminal, _ = env.step(action)
    replay_buffer.append(
      transition(observ, reward, terminal, next_observ, action))
    if terminal:
      observ = env.reset()
    else:
      observ = next_observ
  print('\rFinished initialize memory...')
  return replay_buffer


def main(_):
  init = tf.global_variables_initializer()
  env = gym.make('SpaceInvaders-v0')
  config = utility.AttrDict(default_config())
  with tf.Session() as sess:
    init.run()
    buffer = initialize_memory(sess, env, config)
    print(len(buffer))


if __name__ == '__main__':
  tf.app.run()