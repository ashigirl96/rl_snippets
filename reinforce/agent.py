"""Agent for training environment and eval"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
import tensorflow as tf
from agents.tools.wrappers import ConvertTo32Bit

from reinforce.policy import Policy, ValueFunction
from reinforce.rollout import rollouts

Transition = collections.namedtuple('Transition',
                                    'observ, reward, done, action, next_observ, raw_return, return_')
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


class REINFORCE(object):
    
    def __init__(self, config):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        env = gym.make(config.env_name)
        self.config = config
        self.env = ConvertTo32Bit(env)
        self.policy = Policy(sess=self.sess, config=config)
        self.value_func = ValueFunction(sess=self.sess, config=config)
        
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
        
        policy_losses = []
        value_func_losses = []
        for t, transition in enumerate(trajectory):
            baseline = self.value_func.predict(transition.observ)
            return_ = sum(1. ** i * t.reward for i, t in enumerate(trajectory[t:]))
            advantage = return_ - baseline
            loss_ = self.value_func.apply(transition.observ, return_)
            loss = self.policy.apply(transition.observ, transition.action, advantage)
            policy_losses.append(loss)
            value_func_losses.append(loss_)
        return policy_losses, value_func_losses
    
    def train(self, num_episodes):
        saver = tf.train.Saver()
        for _ in range(num_episodes):
            losses = self._train()
            yield losses
        saver.save(self.sess, './reinforce_debug')