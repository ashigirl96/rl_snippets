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


class ActorCrtic(object):
    
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
        """REINFORCE Algorithm.
    
        Returns: Losses each timestep.
        """
        trajectory = self.compute_trajectory()
        discount_factor = self.config.discount_factor
        
        policy_losses = []
        
        batch_observ, batch_reward, batch_done, batch_action, batch_next_observ, _, _ = zip(*trajectory)
        batch_reward = np.asarray(batch_reward)[..., np.newaxis]
        val_next_observ = np.asarray(self.value_func.predict(batch_next_observ))
        targets = batch_reward + val_next_observ
        val_losses = self.value_func.apply(batch_observ, targets)
        baselines = self.value_func.predict(batch_observ)
        advantages = targets - baselines
        for observ, action, advantage in zip(batch_observ, batch_action, advantages):
            observ = observ[np.newaxis, ...]
            policy_loss = self.policy.apply(observ, action, advantage)
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