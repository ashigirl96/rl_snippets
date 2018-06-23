"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.utils.atari_wrappers import wrap_deepmind

from common import utility
from dqn.configs import default_config
from dqn.networks import ValueFunction

FLAGS = tf.flags.FLAGS


def main(_):
  _config = default_config()
  if _config.use_gpu and len(utility.available_gpus(_config.sess_config)) < 1:
    raise ValueError('There not available gpus...')
  
  if not FLAGS.logdir:
    env = gym.make('SpaceInvaders-v0')
    # env = gym.make('Breakout-v0')
    # env = gym.make('Pong-v0')
    logdir = os.path.abspath('/tmp/{}-{}'.format(
      datetime.datetime.now().strftime('%Y%m%dT%H%M%S'), env.spec.id))
  else:
    logdir = str(FLAGS.logdir)
    env = gym.make(logdir.split('-', 1)[1])
  env = gym.wrappers.Monitor(env, logdir + "/gym", force=True)
  env = wrap_deepmind(env, dim=_config.frame_dim)
  
  atari_actions = np.arange(env.action_space.n, dtype=np.int32)
  
  # Initialize networks.
  with tf.variable_scope('q_network'):
    q_network = ValueFunction(_config,
                              env.observation_space,
                              env.action_space,
                              summaries_dir=logdir)
  with tf.variable_scope('target'):
    target = ValueFunction(_config, env.observation_space, env.action_space, q_network)
  # Epsilon
  eps = np.linspace(_config.epsilon_start, _config.epsilon_end, _config.epsilon_decay_steps)
  
  sess = utility.make_session(_config.sess_config)
  
  import time
  time.sleep(10.)


if __name__ == '__main__':
  tf.flags.DEFINE_string('logdir', '', help='load directory')
  tf.app.run()