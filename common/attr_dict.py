from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib


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