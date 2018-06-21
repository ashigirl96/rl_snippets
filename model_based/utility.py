"""Utility for construct Model-based RL"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import gym
import numpy as np
import tensorflow as tf

from model_based import transition


def make_session(num_cpu=6, graph=None):
  sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    inter_op_parallelism_threads=num_cpu,
    intra_op_parallelism_threads=num_cpu)
  sess_config.gpu_options.allow_growth = True
  return tf.Session(config=sess_config, graph=graph)


def make_environment(use_monitor=False):
  import roboschool
  if roboschool:
    env = gym.make('RoboschoolAnt-v1')
  else:
    env = gym.make('CartPole-v1')
  
  if use_monitor:
    env = gym.wrappers.Monitor(
      env, './logdir/{}'.format(env.spec.id), force=True)
    print('wrapped env {}'.format(env))
  return env


def define_saver(exclude=None) -> tf.train.Saver:
  """Create a saver for the variables we want to checkpoint.
  # https://github.com/tensorflow/agents/

  Args:
    exclude: List of regexes to match variable names to exclude.

  Returns:
    Saver object.
  """
  variables = []
  exclude = exclude or []
  exclude = [re.compile(regex) for regex in exclude]
  for variable in tf.global_variables():
    if any(regex.match(variable.name) for regex in exclude):
      continue
    variables.append(variable)
  saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=5)
  return saver


def initialize_variables(sess, saver, logdir, checkpoint=None, resume=None):
  """Initialize or restore variables from a checkpoint if available.
  # https://github.com/tensorflow/agents/

  Args:
    sess: Session to initialize variables in.
    saver: Saver to restore variables.
    logdir: Directory to search for checkpoints.
    checkpoint: Specify what checkpoint name to use; defaults to most recent.
    resume: Whether to expect recovering a checkpoint or starting a new run.

  Raises:
    ValueError: If resume expected but no log directory specified.
    RuntimeError: If no resume expected but a checkpoint was found.
  """
  sess.run(tf.group(
    tf.local_variables_initializer(),
    tf.global_variables_initializer()))
  if resume and not (logdir or checkpoint):
    raise ValueError('Need to specify logdir to resume a checkpoint.')
  if logdir:
    state = tf.train.get_checkpoint_state(logdir)
    if checkpoint:
      checkpoint = os.path.join(logdir, checkpoint)
      print('checkpoint {}'.format(checkpoint))
    if not checkpoint and state and state.model_checkpoint_path:
      checkpoint = state.model_checkpoint_path
    if checkpoint and resume is False:
      message = 'Found unexpected checkpoint when starting a new run.'
      raise RuntimeError(message)
    print('checkpoint {}'.format(checkpoint))
    if checkpoint:
      saver.restore(sess, checkpoint)


def batch(trajectory, batch_size=1, clip_reward=True, unique=True) -> transition:
  trajectory_size = len(trajectory)
  random_indices = np.random.choice(range(trajectory_size), batch_size, replace=unique)
  observ, action, next_observ, reward, terminal = zip(*np.asarray(trajectory)[random_indices])

  if clip_reward:
    reward = np.sign(reward)
  batch_transition = transition(
    observ=np.asarray(observ),
    action=np.asarray(action),
    next_observ=np.asarray(next_observ),
    reward=np.asarray(reward),
    terminal=np.asarray(terminal),
  )
  return batch_transition