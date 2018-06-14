"""Neural Networks dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np
import roboschool
import tensorflow as tf

transition = collections.namedtuple(
    'transition', 'observ, action, next_observ, reward, terminal')

if roboschool:
    env = gym.make('RoboschoolAnt-v1')
else:
    env = gym.make('CartPole-v1')


def make_session(num_cpu=6):
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    sess_config.gpu_options.allow_growth = True
    return tf.Session(config=sess_config)


def initialize_variables(sess: tf.Session):
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
    ])


class DynamicsNetwork(object):
    
    def __init__(self, env: gym.Env,
                 valid_horization=10):
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError('This rollout is called only continuous ones.')
        if len(env.action_space.shape) > 1:
            raise NotImplementedError('Multi action cannot implemented.')
        if len(env.observation_space.shape) > 1:
            raise NotImplementedError('Multi obsrvation cannot implemented.')
        self._env = env
        self._valid_horization = valid_horization
        network_summary = self._forward()
        loss_summary = self._loss()
        
        self._summaries = tf.summary.merge([
            network_summary,
            loss_summary,
        ])
    
    def predict(self, sess: tf.Session, observ, squeeze=False):
        feed_dict = {self._observ: observ}
        outputs = sess.run(self._outputs, feed_dict=feed_dict)
        if squeeze:
            outputs = np.squeeze(outputs)
        return outputs
    
    def validate(self, sess: tf.Session, observs):
        estimated_observ = observs[0, :]
        h_step_observs = []
        for h in range(self._valid_horization):
            h_step_observs.append(
                estimated_observ + self.predict(sess, estimated_observ[None], True))
        h_step_observs = np.stack(h_step_observs)
        
        feed_dict = {
            self._h_step_estimated_observs: h_step_observs,
            self._h_step_observs: observs[1:(self._valid_horization + 1), :]}
        return sess.run(self._valid_loss, feed_dict=feed_dict)
    
    def _forward(self):
        action_size = self._env.action_space.shape[0]
        self._observ_size = self._env.observation_space.shape[0]
        self._next_observ = tf.placeholder(tf.float32, [None, self._observ_size])
        with tf.variable_scope('dynamics_network'):
            self._observ = tf.placeholder(tf.float32, [None, self._observ_size])
            x = tf.layers.dense(self._observ, 1000, activation=tf.nn.relu)
            x = tf.layers.dense(x, 50, activation=tf.nn.relu)
            self._outputs = tf.layers.dense(x, self._observ_size)
        
        network_summary = tf.summary.merge([
            tf.summary.histogram('outputs', self._outputs),
        ])
        return network_summary
    
    def _loss(self):
        # Train losses.
        losses = tf.losses.mean_squared_error(
            labels=(self._next_observ - self._observ),
            predictions=self._outputs)
        self._loss = tf.reduce_mean(losses)
        
        # Validation losses.
        self._h_step_observs = tf.placeholder(
            tf.float32, (self._valid_horization, self._observ_size))
        self._h_step_estimated_observs = tf.placeholder(
            tf.float32, (self._valid_horization, self._observ_size))
        valid_losses = tf.losses.mean_squared_error(
            labels=self._h_step_observs,
            predictions=self._h_step_estimated_observs,
        )
        self._valid_loss = tf.reduce_mean(valid_losses)
        
        loss_summary = tf.summary.merge([
            tf.summary.histogram('mse_losses', losses),
            tf.summary.scalar('loss', self._loss),
        ])
        return loss_summary


def rollout(policy, env: gym.Env):
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError('This rollout is called only continuous ones.')
    if len(env.action_space.shape) > 1:
        raise NotImplementedError('Multi action cannot impemented.')
    
    action_size = env.action_space.shape[0]
    observ = env.reset()
    
    trajectory = []
    
    for t in itertools.count():
        action_prob = policy.predict(observ)
        action = action_prob.sample()
        assert action.shape == env.action_space.shape
        next_observ, reward, done, _ = env.step(action)
        trajectory.append(transition(observ, action, next_observ, reward, done))
        if done:
            break
    
    # Normalize observations.
    observs, _, next_observs, _, _ = map(np.asarray, zip(*trajectory))
    normalize_observ = np.stack([observs, next_observs], axis=0).mean()
    for i, t in enumerate(trajectory):
        t = t._replace(observ=(t.observ / normalize_observ),
                       next_observ=(t.next_observ / normalize_observ))
        trajectory[i] = t
    
    return trajectory


class RandomPolicy:
    
    def __init__(self, env: gym.Env):
        """Random action policy according environment.
        
        Args:
            env: OpenAI Gym.
        """
        self._env = env
    
    def predict(self, observ):
        """Return random sample distribution."""
        return self._env.action_space


def batch(trajectory):
    observ, action, next_observ, reward, terminal = zip(*np.asarray(trajectory))
    batch_transition = transition(
        observ=np.asarray(observ),
        action=np.asarray(action),
        next_observ=np.asarray(next_observ),
        reward=reward,
        terminal=terminal,
    )
    return batch_transition


def main(_):
    random_policy = RandomPolicy(env)
    trajectory = rollout(random_policy, env)
    batch_transition = batch(trajectory)
    
    dynamics = DynamicsNetwork(env, valid_horization=10)
    
    sess = make_session()
    initialize_variables(sess)
    
    print(dynamics.validate(sess, batch_transition.observ))


if __name__ == '__main__':
    tf.app.run()