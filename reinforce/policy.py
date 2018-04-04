"""Reinforcement learning Policy that deep neural nets given state"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


# class ReinforcePolicy(keras.Model):
#   def __init__(self, num_hiddens1, num_hiddens2, action_space):
#     super(ReinforcePolicy, self).__init__(name='REINFORCE')
#     self.action_space = action_space
#
#     self.dense1 = keras.layers.Dense(num_hiddens1, activation=tf.nn.relu)
#     self.dense2 = keras.layers.Dense(num_hiddens2)
#
#   def call(self, observ, training=None, mask=None):
#     x = self.dense1(observ)
#     x = self.dense2(x)
#     outputs = tf.distributions.Normal(x, 0.01)
#     self.loss = outputs.log_prob()
#     return outputs.sample()
#
#   def compute_action(self, observ):
#     return self.predict(observ)
#
#   def apply_gradients(self, grads_and_vars):
#     pass


class Policy(object):
  
  def __init__(self):
    """Define the layers used, during initialization."""
    self.dense = tf.layers.Dense(10, activation=tf.nn.relu)
    self.output_layer = tf.layers.Dense(1, activation=None)
  
  def predict(self, observ):
    if not isinstance(observ, tf.Tensor):
      observ = tf.constant(observ)
    hidden = self.dense(observ)
    logits = self.output_layer(hidden)
    # return tf.cast(logits > 0., tf.float32)
    return logits


def loss_fn(model: Policy, observ, return_):
  action_prob = tf.nn.softmax(model.predict(observ))
  log_prob = tf.log(action_prob)
  loss = -log_prob * tf.cast(return_, tf.float64)
  return loss


#
# def loss_fn(model: Policy, observ):
#   action_prob = tf.nn.softmax(model.predict(observ))
#   log_prob = -tf.log(action_prob)
#   return log_prob


policy_gradient = tfe.implicit_gradients(loss_fn)