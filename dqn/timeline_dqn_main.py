"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import gym
import numpy as np
import tensorflow as tf

from common import utility
from dqn.configs import default_config
from dqn.memory import initialize_memory, transition
from dqn.networks import ValueFunction

FLAGS = tf.flags.FLAGS


class EpsGreedy(object):
  
  def __init__(self, q_network, action_size):
    self._q_network = q_network
    self._action_size = action_size
    self.eps = tf.Variable(0., False, name='epsilon')
    self._action_prob = tf.Variable(tf.zeros(action_size), False, name='action_prob')
    
    with tf.variable_scope('coefficients'):
      self._summary = tf.summary.merge([
        tf.summary.scalar('eps', self.eps),
        tf.summary.histogram('action_prob', self._action_prob),
      ])
  
  def __call__(self, sess: tf.Session, observ, eps):
    eps_m = eps / self._action_size
    action_prob = np.ones(self._action_size) * eps_m
    q_value = self._q_network.predict(sess, observ[None, ...])
    best_action = np.argmax(q_value)
    action_prob[best_action] += 1 - eps
    
    _ = sess.run([self.eps.assign(eps), self._action_prob.assign(action_prob)])
    summary, global_step = sess.run([self._summary, tf.train.get_global_step()])
    self._q_network._summary_writer.add_summary(
      summary, global_step=global_step)
    return action_prob


class Evaluate(object):
  def __init__(self, sess: tf.Session, env: gym.Env, q_network: ValueFunction):
    self._sess = sess
    self._env = env
    self._q_network = q_network
    self._mean_episode_reward = tf.Variable(0., False, name='mean_episode_reward')
    self._best_mean_episode_reward = tf.Variable(0., False, name='best_mean_episode_reward')
    
    self._assign_mean = tf.no_op()
    self._assign_best = tf.no_op()
    
    with tf.variable_scope('evaluate_return'):
      self._summary = tf.summary.merge([
        tf.summary.scalar(self._mean_episode_reward.name,
                          self._mean_episode_reward),
        tf.summary.scalar(self._best_mean_episode_reward.name,
                          self._best_mean_episode_reward),
      ])
  
  def evaluate(self):
    episode_rewards = utility.get_wrapper_by_name(
      self._env, "Monitor").get_episode_rewards()
    if len(episode_rewards) > 0:
      self._assign_mean = self._mean_episode_reward.assign(
        np.mean(episode_rewards[-100:]))
    if len(episode_rewards) > 100:
      self._assign_best = self._best_mean_episode_reward.assign(
        tf.reduce_max([self._best_mean_episode_reward, self._mean_episode_reward]))
    
    _, summary, global_step = self._sess.run([
      [self._assign_mean, self._assign_best],
      self._summary,
      tf.train.get_global_step(),
    ])
    self._q_network._summary_writer.add_summary(
      summary, global_step=global_step)
    print(' ... ', end='')
    return episode_rewards[-1], len(episode_rewards)


def main(_):
  _config = default_config()
  if _config.use_gpu and len(utility.available_gpus(_config.sess_config)) < 1:
    raise ValueError('There not available gpus...')
  env, logdir = utility.make_atari_environment(FLAGS.logdir, _config.frame_dim, use_monitor=_config.use_monitor)
  atari_actions = np.arange(env.action_space.n, dtype=np.int32)
  
  # Initialize networks.
  with tf.variable_scope('q_network'):
    q_network = ValueFunction(_config,
                              env.observation_space,
                              env.action_space,
                              summaries_dir=logdir)
  with tf.variable_scope('target'):
    target = ValueFunction(_config, env.observation_space, env.action_space, q_network)
  # Epsilon
  eps = np.linspace(_config.epsilon_start, _config.epsilon_end, _config.epsilon_decay_steps)
  policy = EpsGreedy(q_network, env.action_space.n)
  
  saver = utility.define_saver(exclude=('.*_temporary.*',))
  sess = utility.make_session(_config.sess_config)
  evaluaor = Evaluate(sess, env, q_network)
  utility.initialize_variables(sess, saver, logdir)
  
  # Initialize memory
  memory = initialize_memory(sess, env, _config, q_network)
  # Initialize policy
  total_step = sess.run(tf.train.get_global_step())
  last_timestep = 0
  print('total_step', total_step)
  
  for episode in range(_config.num_episodes):
    filename = os.path.join(logdir, 'model.ckpt')
    saver.save(sess, filename, global_step=tf.train.get_global_step())
    observ = env.reset()
    for t in itertools.count():
      action_prob = policy(sess, observ,
                           eps[min(total_step, _config.epsilon_decay_steps - 1)])
      action = np.random.choice(atari_actions, size=1, p=action_prob)[0]
      next_observ, reward, terminal, _ = env.step(action)
      memory.append(
        transition(observ, reward, terminal, next_observ, action))
      
      batch_transition = memory.sample(_config.batch_size)
      best_actions = q_network.best_action(sess, batch_transition.next_observ)
      target_values = target.estimate(sess,
                                      batch_transition.reward,
                                      batch_transition.terminal,
                                      batch_transition.next_observ,
                                      best_actions)
      
      loss, update_duration = utility.calculate_duration(
        lambda: q_network.update_step(sess, batch_transition.observ,
                                      batch_transition.action,
                                      target_values, total_step))
      print('\r({}/{}) loss: {} update duration {}'.format(
        total_step, _config.max_total_step_size, loss, update_duration), end='', flush=True)
      
      if total_step % _config.update_target_estimator_every == 0:
        print('\nUpdate Target Network...')
        target.assign(sess)
      
      if terminal:
        break
      
      observ = next_observ
      total_step += 1
    last_rewards, length_ = evaluaor.evaluate()
    print('Episode ({}/{}), last return: {}, last timesteps: {:05}'.format(
      episode, _config.num_episodes, last_rewards, total_step - last_timestep))


if __name__ == '__main__':
  tf.flags.DEFINE_string('logdir', '', help='load directory')
  tf.app.run()