"""Train the Dynamics and Controller."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import tensorflow as tf

from common import utility
from model_based.dynamics import DynamicsNetwork
from model_based.policy import RandomPolicy
from model_based.rollouts import Experiment


class default_config:
  max_episodes = 10_000
  timesteps = int(1e8)
  
  training_epochs = 60
  aggregation_iters = 0
  train_epochs_per_agg_iters = 60


def train():
  tf.reset_default_graph()
  env = utility.make_environment(use_monitor=True)
  random_policy = RandomPolicy(env)
  experiment = Experiment(env)
  dynamics = DynamicsNetwork(env, valid_horizon=10)
  config = default_config()
  
  random_dataset = [experiment.rollout(random_policy, True) for _ in range(100)]
  rl_dataset = []
  
  saver = utility.define_saver(exclude=('.*_temporary.*',))
  sess = utility.make_session()
  utility.initialize_variables(sess, saver, './logdir')
  
  for i_episode in itertools.count():
    for trajectory in [*random_dataset, *rl_dataset]:
      batch_transition = utility.batch(trajectory)
      dynamics.update(sess,
                      batch_transition.observ,
                      batch_transition.next_observ,
                      batch_transition.action)
    
    for t_timestep in itertools.count():
      # experiment.rollout ...
      
      if t_timestep >= config.timesteps:
        break
    
    if i_episode >= config.max_episodes:
      break


def main(_):
  pass


if __name__ == '__main__':
  tf.app.run()