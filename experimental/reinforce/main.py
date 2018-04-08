"""One file REINFORCE algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np
import tensorflow as tf
from agents.tools.attr_dict import AttrDict
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.utils.filter import MeanStdFilter

Timestep = collections.namedtuple('Timestep',
                                  'observ, reward, done, action, next_observ, raw_return, return_')
Timestep.__new__.__defaults__ = (None,) * len(Timestep._fields)


class Policy(object):
    
    def __init__(self, config):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        self.config = config
        self._build_model()
        self._set_loss()
        self.varialbes = TensorFlowVariables(self.loss, self.sess)
        
        self.sess.run(tf.global_variables_initializer())
    
    def _build_model(self):
        self.observ = tf.placeholder(tf.float32, (None, 2))
        self.action = tf.placeholder(tf.float32, (None, 1))
        x = tf.layers.dense(self.observ, 100, use_bias=False)
        x = tf.layers.dense(x, 100, use_bias=False)
        x = tf.layers.dense(x, 1, use_bias=False)
        x = tf.clip_by_value(x, -1., 1.)
        self.model = x
    
    def _set_loss(self):
        prob = tf.nn.softmax(self.model)
        prob = tf.exp(prob) - tf.exp(self.action)
        log_prob = -tf.log(prob)
        log_prob = tf.check_numerics(log_prob, 'log_prob')
        self.loss = log_prob
    
    def compute_action(self, observ):
        observ = observ[np.newaxis, ...]
        assert observ.shape == (1, 2)
        action = self.sess.run(self.model, feed_dict={self.observ: observ})
        return action[0]


def rollouts(env: gym.Env, policy: Policy, reward_filter: MeanStdFilter, config):
    raw_return = 0
    return_ = 0
    observ = env.reset()
    
    trajectory = []
    for t in itertools.count():
        action = policy.compute_action(observ)
        
        next_observ, reward, done, _ = env.step(action)
        
        raw_return += reward
        return_ += reward * config.discount_factor ** t
        
        trajectory.append(Timestep(observ, reward, done, action, next_observ, raw_return, return_))
        if done:
            break
    return trajectory


def default_config():
    use_bias = False
    env_name = 'MountainCarContinuous-v0'
    discount_factor = 0.995
    
    return locals()


def main(_):
    reward_filter = MeanStdFilter((), clip=5.)
    config = AttrDict(default_config())
    env = gym.make(config.env_name)
    policy = Policy(config)
    
    traject = rollouts(env, policy, reward_filter, config)
    print(traject[0])


if __name__ == '__main__':
    tf.app.run()