"""Agent for training environment and eval"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import ray
import tensorflow as tf
from agents.tools.wrappers import ConvertTo32Bit

import gym
from general_reinforce.policy import Policy, ValueFunction
from general_reinforce.rollout import rollouts

Transition = collections.namedtuple('Transition',
                                    'observ, reward, done, action, next_observ, raw_return, return_')
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)

TrainResult = collections.namedtuple('TrainResult', 'policy_loss, val_loss, eval')
TrainResult.__new__.__defaults__ = (None,) * len(TrainResult._fields)


class REINFORCE(object):
    
    def __init__(self, config, env_info):
        tf.reset_default_graph()
        self.sess = tf.Session(config=config.sess_config)
        
        env = gym.make(config.env_name)
        self.config = config
        self.env_info = env_info
        self.env = ConvertTo32Bit(env)
        with tf.device('/cpu:0'):
            self.policy = Policy(config=config, env_info=env_info, sess=self.sess)
            self.value_func = ValueFunction(config=config, env_info=env_info, sess=self.sess)
        self._init()
    
    def _init(self):
        self.sess.run(tf.global_variables_initializer())
    
    def compute_trajectory(self):
        trajectory = rollouts(self.env,
                              self.policy)
        return trajectory
    
    def _train(self):
        """REINFORCE Algorithm.
    
        Returns: Losses each timestep.
        """
        trajectory = self.compute_trajectory()
        discount_factor = self.config.discount_factor
        
        policy_losses = []
        val_losses = []
        for t, transition in enumerate(trajectory):
            baseline = self.value_func.predict(transition.observ)
            return_ = sum(discount_factor ** i * t_.reward for i, t_ in enumerate(trajectory[t:]))
            # Compute advantage.
            advantage = return_ - baseline
            # Apply value function gradients.
            loss_ = self.value_func.apply(transition.observ, return_)
            # Apply policy gradients.
            loss = self.policy.apply(transition.observ, transition.action, advantage)
            # loss = self.policy.apply(transition.observ, transition.action, return_)
            policy_losses.append(loss)
            val_losses.append(loss_)
        return policy_losses, val_losses
    
    def train(self, num_episodes):
        """Train the agents.
        Args:
            num_episodes: the number of episodes

        Yields:
            Training Result mainly losses.
        """
        saver = tf.train.Saver()
        for _ in range(num_episodes):
            policy_losses, val_losses = self._train()
            policy_loss = np.mean(policy_losses)
            val_loss = np.mean(val_losses)
            yield TrainResult(policy_loss, val_loss)
        saver.save(self.sess, './checkpoints/reinforce_debug')
    
    def evaluate(self, remote_policies):
        """Evaluate agent to use remote policies.
        
        Args:
            remote_policies:

        Returns:

        """
        weights = self.policy.get_weights()
        weights_id = ray.put(weights)
        
        evaluate_ids = [remote_policies[i].evaluate.remote(weights_id) for i in range(5)]
        evals = ray.get(evaluate_ids)
        return evals