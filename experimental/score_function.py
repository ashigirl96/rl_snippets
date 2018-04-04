"""Experimental how to compute gradients in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


def compute_z(mu):
  eps = tf.distributions.Normal(0., 1.)
  z = mu + eps.stddev() * eps.sample()
  return z


def grad(q: tf.distributions.Distribution, mu):
  with tfe.GradientTape() as tape:
    z = compute_z(mu)
    prob = q.log_prob(tf.stop_gradient(z))
  return tape.gradient(prob, [mu])


def main(_):
  mu = tfe.Variable(5.)
  q = tf.distributions.Normal(mu, 1.)
  gradients = grad(q, mu)
  print(gradients)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run()