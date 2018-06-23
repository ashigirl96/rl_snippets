"""Utility for construct Model-based RL"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import re
import time

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.utils.atari_wrappers import wrap_deepmind

from model_based.rollouts import transition


def calculate_duration(process_fn):
  start_time = time.time()
  return_ = process_fn()
  end_time = time.time()
  duration = end_time - start_time
  return return_, duration


def trainable_variables(network, sorted=True):
  scope_name = network.value.name.split('/')[0]
  vars_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
  if sorted:
    vars_.sort(key=lambda x: x.name)
  return vars_


# TODO: args should be loss??
def compute_gradient(network):
  vars_ = trainable_variables(network)
  with tf.name_scope('network_gradients'):
    gradients = tf.gradients(network.value, vars_)
  print(gradients)


def available_gpus(sess_config):
  from tensorflow.python.client import device_lib
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices(session_config=sess_config)
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def make_sess_config(num_cpu=6):
  sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    inter_op_parallelism_threads=num_cpu,
    intra_op_parallelism_threads=num_cpu)
  sess_config.gpu_options.allow_growth = True
  return sess_config


def make_session(sess_config, graph=None):
  return tf.Session(config=sess_config, graph=graph)


class ClipAction(object):
  """Clip out of range actions to the action space of the environment."""
  
  def __init__(self, env):
    self._env = env
  
  def __getattr__(self, name):
    return getattr(self._env, name)
  
  @property
  def action_space(self):
    shape = self._env.action_space.shape
    low, high = -1. * np.ones(shape), 1. * np.ones(shape)
    return gym.spaces.Box(low, high, dtype=np.float32)
  
  def step(self, action):
    action_space = self._env.action_space
    action = np.clip(action, action_space.low, action_space.high)
    return self._env.step(action)


def make_environment(use_monitor=False):
  from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
  env = HalfCheetahEnv()
  env = ClipAction(env)
  
  if use_monitor:
    env = gym.wrappers.Monitor(
      env, './logdir/{}'.format(env.spec.id), force=True)
    print('wrapped env {}'.format(env))
  return env


def make_atari_environment(logdir, frame_dim, use_monitor=True) -> (gym.Env, str):
  if not logdir:
    env = gym.make('SpaceInvaders-v0')
    # env = gym.make('Breakout-v0')
    # env = gym.make('Pong-v0')
    logdir = os.path.abspath('/tmp/{}-{}'.format(
      datetime.datetime.now().strftime('%Y%m%dT%H%M%S'), env.spec.id))
  else:
    logdir = str(logdir)
    env = gym.make(logdir.split('-', 1)[1])
  if use_monitor:
    env = gym.wrappers.Monitor(env, logdir + "/gym", force=True)
  env = wrap_deepmind(env, dim=frame_dim)
  return env, logdir


def get_wrapper_by_name(env, classname):
  currentenv = env
  while True:
    if classname in currentenv.__class__.__name__:
      return currentenv
    elif isinstance(env, gym.Wrapper):
      currentenv = currentenv.env
    else:
      raise ValueError("Couldn't find wrapper named %s" % classname)


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