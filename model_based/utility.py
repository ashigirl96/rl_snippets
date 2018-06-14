"""Utility for construct Model-based RL"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def initialize_variables(sess: tf.Session):
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
    ])


def batch(trajectory):
    observ, action, next_observ, reward, terminal = zip(*np.asarray(trajectory))
    batch_transition = transition(
        observ=np.asarray(observ),
        action=np.asarray(action),
        next_observ=np.asarray(next_observ),
        reward=reward,
        terminal=terminal,
    )
    return batch_transition