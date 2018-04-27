"""Agent for training environment and eval"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
import numpy as np
import ray
import tensorflow as tf
from agents.tools.wrappers import ConvertTo32Bit

from actor_critic.policy import Policy, ValueFunction
from actor_critic.rollout import rollouts

Transition = collections.namedtuple('Transition',
                                    'observ, reward, done, action, next_observ, raw_return, return_')
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)

TrainResult = collections.namedtuple('TrainResult', 'policy_loss, val_loss, eval')
TrainResult.__new__.__defaults__ = (None,) * len(TrainResult._fields)


class OnlineActorCrtic(object):
    
    def __init__(self, config):
        tf.reset_default_graph()
        self.sess = tf.Session(config=config.sess_config)
        
        env = gym.make(config.env_name)
        self.config = config
        self.env = ConvertTo32Bit(env)
        with tf.device('/cpu:0'):
            self.policy = Policy(config=config, sess=self.sess)
            self.value_func = ValueFunction(config=config, sess=self.sess)
        self._init()
    
    def _init(self):
        self.sess.run(tf.global_variables_initializer())
    
    def compute_trajectory(self):
        trajectory = rollouts(self.env,
                              self.policy)
        return trajectory
    
    def _train(self):
        """Actor Critic Algorithm.
    
        Returns: Losses each timestep.
        """
        trajectory = self.compute_trajectory()
        discount_factor = self.config.discount_factor
        
        policy_losses = []
        val_losses = []
        
        for t, transition in enumerate(trajectory):
            observ = transition.observ[np.newaxis, ...]
            next_observ = transition.next_observ[np.newaxis, ...]
            # return_ = sum(discount_factor ** i * t_.reward for i, t_ in enumerate(trajectory[t:]))
            
            val_next_observ = self.value_func.predict(next_observ)
            td_target = transition.reward + discount_factor * val_next_observ
            baseline = self.value_func.predict(observ)
            td_error = td_target - baseline

            val_loss = self.value_func.apply(observ, td_target)
            policy_loss = self.policy.apply(observ, transition.action, td_error)
            
            val_losses.append(val_loss)
            policy_losses.append(policy_loss)
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