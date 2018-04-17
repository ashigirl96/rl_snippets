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
    return tf.constant(return_, dtype=tf.float64)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print('{0}{1}{2}'.format(bcolors.OKGREEN, get_available_gpus(), bcolors.ENDC))