"""Rollout one episode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import gym
import numpy as np
import ray
import tensorflow as tf
from tensorflow import keras
from ray.rllib.ppo import PPOAgent

from reinforce.policy import Policy
from reinforce.agent import Agent

ray.init()
ray.error_info()

Timestep = namedtuple('Timestep', ['observ', 'action', 'reward'])


def rollouts(env, policy: Policy):
  """
  EXERCISE:
    Fill out this function by copying the 'random_rollout' function
    and then modifying it to choose the action using the policy.
  
  Args:
    env: Environment of OpenAI Gym
    policy: Agent's Policy

  Returns:
    cumulative_reward: Discounted episodic return
  """
  observ = env.reset()
  observ = observ[np.newaxis, ...]
  
  done = False
  cumulative_reward = 0
  
  trajectory = list()
  
  while not done:
    # Choose a random action (either 0 or 1)
    action = tf.cast(policy.predict(observ) > 0., tf.float32)
    action = tf.squeeze(action)
    action = int(action.numpy())
    
    # Take the action in the env.
    observ_, reward, done, _ = env.step(action)
    observ_ = observ_[np.newaxis, ...]
    
    # Update the cumulative reward.
    cumulative_reward += reward
    trajectory.append(Timestep(observ, action, reward))
    
    observ = observ_
    
  # Return the cumulative reward.
  return trajectory


@ray.remote
def evaluate_policy(num_rollouts):
  # Generate a random policy.
  policy = Policy()
  
  # Create an environment.
  env = gym.make('CartPole-v0')
  
  # Evaluate the same policy multiple times
  # and then take the average in order to evaluate the policy more accurately
  trajectory = rollouts(env, policy)
  
  


def main():
  tf.enable_eager_execution()


if __name__ == '__main__':
  main()