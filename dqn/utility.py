"""Utility for construct DQN model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import tensorflow as tf
import dqn.nested as nested


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


def available_gpus():
  from tensorflow.python.client import device_lib
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def linear_interpolation(l, r, alpha):
  return l + alpha * (r - l)


class PiecewiseSchedule(object):
  def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
    """Piecewise schedule.
    endpoints: [(int, int)]
        list of pairs `(time, value)` meanining that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    interpolation: lambda float, float, float: float
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    outside_value: float
        if the value is requested outside of all the intervals sepecified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    """
    idxes = [e[0] for e in endpoints]
    assert idxes == sorted(idxes)
    self._interpolation = interpolation
    self._outside_value = outside_value
    self._endpoints = endpoints
  
  def value(self, t):
    """See Schedule.value"""
    for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
      if l_t <= t and t < r_t:
        alpha = float(t - l_t) / (r_t - l_t)
        return self._interpolation(l, r, alpha)
    
    # t does not belong to any of the pieces, so doom.
    assert self._outside_value is not None
    return self._outside_value


class AttrDict(dict):
  """Wrap a dictionary to access keys as attributes."""
  
  # https://github.com/tensorflow/agents/
  
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    super(AttrDict, self).__setattr__('_mutable', False)
  
  def __getattr__(self, key):
    # Do not provide None for unimplemented magic attributes.
    if key.startswith('__'):
      raise AttributeError
    return self.get(key, None)
  
  def __setattr__(self, key, value):
    if not self._mutable:
      message = "Cannot set attribute '{}'.".format(key)
      message += " Use 'with obj.unlocked:' scope to set attributes."
      raise RuntimeError(message)
    if key.startswith('__'):
      raise AttributeError("Cannot set magic attribute '{}'".format(key))
    self[key] = value
  
  @property
  @contextlib.contextmanager
  def unlocked(self):
    super(AttrDict, self).__setattr__('_mutable', True)
    yield
    super(AttrDict, self).__setattr__('_mutable', False)
  
  def copy(self):
    return type(self)(super(AttrDict, self).copy())