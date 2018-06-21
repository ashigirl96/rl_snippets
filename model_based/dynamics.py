"""Neural Networks dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from model_based import utility
from model_based.policy import RandomPolicy
from model_based.rollouts import Experiment


class DynamicsNetwork(object):
  
  def __init__(self, env: gym.Env,
               valid_horization=10,
               log_dir='./logdir'):
    if not isinstance(env.action_space, gym.spaces.Box):
      raise ValueError('This rollout is called only continuous ones.')
    if len(env.action_space.shape) > 1:
      raise NotImplementedError('Multi action cannot implemented.')
    if len(env.observation_space.shape) > 1:
      raise NotImplementedError('Multi obsrvation cannot implemented.')
    self._env = env
    self._valid_horization = valid_horization
    self._log_dir = log_dir
    
    network_summary = self._forward()
    train_summary, valid_summary = self._loss()
    self._optimizer()
    self._summary()
    
    self._train_summaries = tf.summary.merge([
      network_summary,
      train_summary,
    ])
    self._valid_summaries = tf.summary.merge([
      valid_summary,
    ])
  
  def predict(self, sess: tf.Session, observ, action, squeeze=False):
    if observ.ndim == 1:
      observ = observ[None]
    if action.ndim == 1:
      action = action[None]
    feed_dict = {self._observ: observ, self._action: action}
    outputs = sess.run(self._outputs, feed_dict=feed_dict)
    if squeeze:
      outputs = np.squeeze(outputs)
    return outputs
  
  def update(self, sess: tf.Session, batch_observ, batch_next_observ, batch_action):
    feed_dict = {
      self._observ: batch_observ,
      self._next_observ: batch_next_observ,
      self._action: batch_action,
    }
    _, loss, global_step, summary = sess.run(
      [self._train_op, self._train_loss, tf.train.get_global_step(), self._train_summaries],
      feed_dict=feed_dict)
    self._train_summary_writer.add_summary(summary, global_step=global_step)
    return loss
  
  def validate(self, sess: tf.Session, observs, actions):
    estimated_observ = observs[0, :]
    h_step_estimated_observs = []
    for h in range(self._valid_horization):
      h_step_estimated_observs.append(
        estimated_observ + self.predict(
          sess, estimated_observ, actions[h, :], squeeze=True))
    h_step_estimated_observs = np.stack(h_step_estimated_observs)
    
    feed_dict = {
      self._h_step_estimated_observs: h_step_estimated_observs,
      self._h_step_observs: observs[1:(self._valid_horization + 1), :]}
    loss, global_step, summary = sess.run(
      [self._valid_loss, tf.train.get_global_step(), self._valid_summaries],
      feed_dict=feed_dict)
    self._valid_summary_writer.add_summary(summary, global_step=global_step)
    return loss
  
  def _forward(self):
    action_size = self._env.action_space.shape[0]
    self._observ_size = self._env.observation_space.shape[0]
    self._next_observ = tf.placeholder(tf.float32, [None, self._observ_size])
    with tf.variable_scope('forward_dynamics_network'):
      self._observ = tf.placeholder(tf.float32, [None, self._observ_size])
      self._action = tf.placeholder(tf.float32, [None, action_size])
      x = tf.concat([self._observ, self._action], axis=1)
      x = tf.layers.dense(x, 1000, activation=tf.nn.relu)
      x = tf.layers.dense(x, 50, activation=tf.nn.relu)
      self._outputs = tf.layers.dense(x, self._observ_size)
    
    network_summary = tf.summary.merge([
      tf.summary.histogram('outputs', self._outputs),
    ])
    return network_summary
  
  def _loss(self):
    # Train losses.
    with tf.name_scope('train_losses'):
      losses = tf.losses.mean_squared_error(
        labels=(self._next_observ - self._observ),
        predictions=self._outputs)
      self._train_loss = tf.reduce_mean(losses)
    
    # Validation losses.
    with tf.name_scope('validation_losses'):
      self._h_step_observs = tf.placeholder(
        tf.float32, (self._valid_horization, self._observ_size))
      self._h_step_estimated_observs = tf.placeholder(
        tf.float32, (self._valid_horization, self._observ_size))
      valid_losses = tf.losses.mean_squared_error(
        labels=self._h_step_observs,
        predictions=self._h_step_estimated_observs,
      )
      self._valid_loss = tf.reduce_mean(valid_losses)
    
    train_summary = tf.summary.merge([
      tf.summary.histogram('mse_losses', losses),
      tf.summary.scalar('loss', self._train_loss),
    ])
    valid_summary = tf.summary.merge([
      tf.summary.histogram('valid_losses', valid_losses),
      tf.summary.scalar('valid_loss', self._valid_loss),
    ])
    return train_summary, valid_summary
  
  def _optimizer(self):
    with tf.name_scope('optimizer'):
      optimizer = tf.train.AdamOptimizer(0.0001)
      self._train_op = optimizer.minimize(
        self._train_loss,
        global_step=tf.train.get_or_create_global_step())
  
  def _summary(self, graph=None):
    if not graph:
      graph = tf.get_default_graph()
    train_dir = self._log_dir + '/training'
    valid_dir = self._log_dir + '/validation'
    self._train_summary_writer = tf.summary.FileWriter(train_dir, graph=graph)
    self._valid_summary_writer = tf.summary.FileWriter(valid_dir)


def main(_):
  tf.reset_default_graph()
  env = utility.make_environment(use_monitor=False)
  random_policy = RandomPolicy(env)
  experiment = Experiment(env)
  trajectory = experiment.rollout(random_policy)
  batch_transition = utility.batch(trajectory, batch_size=32)
  print(batch_transition.observ.shape)
  
  dynamics = DynamicsNetwork(env, valid_horization=10)

  saver = utility.define_saver(exclude=('.*_temporary.*',))
  sess = utility.make_session()
  utility.initialize_variables(sess, saver, './logdir')

  print(dynamics.validate(sess, batch_transition.observ, batch_transition.action))
  print(dynamics.predict(sess,
                         observ=batch_transition.observ,
                         action=batch_transition.action,
                         squeeze=True).shape)
  print(dynamics.update(sess,
                        batch_observ=batch_transition.observ,
                        batch_action=batch_transition.action,
                        batch_next_observ=batch_transition.next_observ))
  saver.save(sess, './logdir/dynamics')


if __name__ == '__main__':
  tf.app.run()