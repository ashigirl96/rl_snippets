from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import numpy as np
import os
import ray
import tensorflow as tf
from ray_tutorial.reinforce.env import BatchedEnv
from ray_tutorial.reinforce.env import (NoPreprocessor)
from ray_tutorial.reinforce.filter import RunningStat
from ray_tutorial.reinforce.policy import ProximalPolicyLoss
from ray_tutorial.reinforce.rollout import add_advantage_values, rollouts


class AttrDict(dict):
    """Wrap a dictionary to access keys as attributes."""
    
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        super(AttrDict, self).__setattr__('_mutable', False)
    
    def __getattr__(self, key):
        # Do not provide None for unimplemented magic attributes.
        if key.startswith('__'):
            raise AttributeError
        return self.get(key, None)
    
    def __setattr__(self, key, value):
        if not self._mutable:
            message = "Cannot set attribute '{}'.".format(key)
            message += " Use 'with obj.unlocked:' scope to set attributes."
            raise RuntimeError(message)
        if key.startswith('__'):
            raise AttributeError("Cannot set magic attribute '{}'".format(key))
        self[key] = value
    
    @property
    @contextlib.contextmanager
    def unlocked(self):
        super(AttrDict, self).__setattr__('_mutable', True)
        yield
        super(AttrDict, self).__setattr__('_mutable', False)


class MeanStdFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        
        self.rs = RunningStat(shape)
    
    def __call__(self, x, update=True):
        x = np.asarray(x)
        if update:
            if len(x.shape) == len(self.rs.shape) + 1:
                # The vectorized case.
                for i in range(x.shape[0]):
                    self.rs.push(x[i])
            else:
                # The unvectorized case.
                self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            # if np.amin(x) < -self.clip or np.amax(x) > self.clip:
            #     print("Clipping value to " + str(self.clip))
            x = np.clip(x, -self.clip, self.clip)
        return x


class Agent(object):
    
    def __init__(self, name, batchsize, preprocessor, config, use_gpu):
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.env = BatchedEnv(name, batchsize, preprocessor=preprocessor)
        if preprocessor.shape is None:
            preprocessor.shape = self.env.observation_space.shape
        self.sess = tf.Session()
        self.ppo = ProximalPolicyLoss(self.env.observation_space, self.env.action_space, preprocessor, config,
                                      self.sess)
        self.optimizer = tf.train.AdamOptimizer(config["sgd_stepsize"])
        self.train_op = self.optimizer.minimize(self.ppo.loss)
        self.variables = ray.experimental.TensorFlowVariables(self.ppo.loss, self.sess)
        self.observation_filter = MeanStdFilter(preprocessor.shape, clip=None)
        self.reward_filter = MeanStdFilter((), clip=5.0)
        self.config = config
        self.sess.run(tf.global_variables_initializer())
    
    def get_weights(self):
        return self.variables.get_weights()
    
    def load_weights(self, weights):
        self.variables.set_weights(weights)
    
    def compute_trajectory(self, gamma, lam, horizon):
        trajectory = rollouts(self.ppo, self.env, horizon, self.observation_filter, self.reward_filter)
        add_advantage_values(trajectory, gamma, lam, self.reward_filter)
        return trajectory
    
    def _train(self, prev_logits, kl_coeff, observations, advantages, actions):
        trajectory = {
            self.ppo.prev_logits:  prev_logits,
            self.ppo.kl_coeff:     kl_coeff,
            self.ppo.observations: observations,
            self.ppo.advantages:   advantages,
            self.ppo.actions:      actions,
        }
        _, loss = self.sess.run([self.train_op, self.ppo.loss], feed_dict=trajectory)
        return loss
    
    def train(self, num_iter):
        for _ in range(num_iter):
            trajectory = AttrDict(self.compute_trajectory(
                self.config.gamma, self.config.lam, self.config.horizon))
            
            with trajectory.unlocked:
                trajectory.logprobs = np.mean(trajectory.logprobs, 0)
                trajectory.observations = np.mean(trajectory.observations, 0)
                trajectory.advantages = np.mean(trajectory.advantages, 0)
                trajectory.actions = np.mean(trajectory.actions, 0).squeeze()
            
            loss = self._train(kl_coeff=self.config.kl_coeff,
                               observations=trajectory.observations,
                               advantages=trajectory.advantages,
                               actions=trajectory.actions,
                               prev_logits=trajectory.logprobs,
                               )
            yield loss


def default_config():
    kl_coeff = 0.2
    num_sgd_iter = 30
    sgd_stepsize = 5e-5
    sgd_batchsize = 128
    entropy_coeff = 0.0
    clip_param = 0.3
    kl_target = 0.01
    timesteps_per_batch = 40000
    
    name = "CartPole-v0"
    batchsize = 100
    preprocessor = NoPreprocessor()
    gamma = 0.995
    lam = 1.0
    horizon = 2000
    
    return locals()


def main(_):
    config = AttrDict(default_config())
    
    agent = Agent(config.name, config.batchsize, config.preprocessor, config, use_gpu=True)
    
    for loss in agent.train(1_000):
        print('loss: {0}'.format(loss))


if __name__ == '__main__':
    # ray.init(num_cpus=4, redirect_output=True)
    tf.app.run()