"""Test Utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common import utility
from model_based.policy import RandomPolicy
from model_based.rollouts import Experiment
import numpy as np


class TestUtility(tf.test.TestCase):
  def test_batch_size(self):
    tf.reset_default_graph()
    env = utility.make_environment(use_monitor=False)
    random_policy = RandomPolicy(env)
    experiment = Experiment(env)
    trajectory = experiment.rollout(random_policy)
    batch_transition = utility.batch(trajectory, batch_size=1)
    print(batch_transition.observ)
    print(batch_transition.action)
    print(batch_transition.reward)
    print(batch_transition.terminal)
    self.assertEqual(batch_transition.observ.shape, (1, 28))

    batch_transition = utility.batch(trajectory, batch_size=32)
    self.assertEqual(batch_transition.observ.shape, (32, 28))


  def test_clip_reward(self):
    tf.reset_default_graph()
    env = utility.make_environment(use_monitor=False)
    random_policy = RandomPolicy(env)
    experiment = Experiment(env)
    trajectory = experiment.rollout(random_policy)
    batch_transition = utility.batch(trajectory, batch_size=32)
    batch_reward = batch_transition.reward
    self.assertEqual(np.max(batch_reward), 1.)
    self.assertEqual(np.min(batch_reward), -1.)

if __name__ == '__main__':
  tf.test.main()

