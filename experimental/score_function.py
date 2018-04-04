"""Experimental how to compute gradients in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


class Policy(object):
  
  def __init__(self):
    """Define the layers used, during initialization."""
    self.dense = tf.layers.Dense(10, activation=tf.nn.relu)
    self.output_layer = tf.layers.Dense(2, activation=None)
  
  def predict(self, observ):
    hidden = self.dense(observ)
    logits = self.output_layer(hidden)
    return tf.nn.softmax(logits)


def loss_fn(model: Policy, observ):
  action_prob = model.predict(observ)
  log_prob = -tf.log(action_prob)
  return log_prob


def main(_):
  policy = Policy()
  optimizer = tf.train.GradientDescentOptimizer(5e-1)
  policy_gradient = tfe.implicit_gradients(loss_fn)
  
  env = gym.make('CartPole-v0')
  observs = [env.reset()]
  for i in range(5):
    observs.append(env.step(env.action_space.sample())[0])
  observs = tf.constant(tf.stack(observs))
  optimizer.apply_gradients(policy_gradient(policy, observs))
  print(loss_fn(policy, observs))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run()