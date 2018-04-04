"""Utility function and class for implement RENFORCE"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def discounted_return(discount_factor, rewards):
  """Calculate discounted episodic return(total reward)
  
  Args:
    discount_factor: discount the return for devergence infinity.
    trajectory: rewards begin from timestep t.

  Returns:
    calculated discounted value.
  """
  return_ = sum([reward * discount_factor ** l for l, reward in enumerate(rewards)])
  return tf.constant(return_)