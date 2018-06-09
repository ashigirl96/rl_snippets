"""Utility for construct DQN model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def trainable_variables(network, sorted=True):
    scope_name = network.value.name.split('/')[0]
    vars_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    if sorted:
        vars_.sort(key=lambda x: x.name)
    return vars_


#TODO: args should be loss??
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
