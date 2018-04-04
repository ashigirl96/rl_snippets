"""main file coded the reinforce algorithm.
I will replace this code to reusability."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
from agents.tools import AttrDict

from reinforce.policy import Policy, policy_gradient
from reinforce.rollout import rollouts
from reinforce.utils import discounted_return


def default_config():
  num_episodes = 10
  num_iters = 5
  lr = 0.001
  discount_factor = 0.99
  
  return locals()


def main(_):
  config = AttrDict(default_config())
  
  policy = Policy()
  optimizer = tf.train.GradientDescentOptimizer(5e-1)
  
  env = gym.make('CartPole-v0')
  
  for _ in range(config.num_iters):
    trajectories = []
    for i in range(config.num_episodes):
      trajectory = rollouts(env, policy)
      trajectories.append(trajectory)
    
    for trajectory in trajectories:
      for t, timestep in enumerate(trajectory):
        rewards = [traj.reward for traj in trajectory[t:]]
        return_ = discounted_return(config.discount_factor, rewards)
        # grad = policy_gradient(policy, timestep.observ, return_)
        optimizer.apply_gradients(policy_gradient(policy, timestep.observ, return_))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run()