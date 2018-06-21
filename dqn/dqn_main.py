"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.utils.atari_wrappers import wrap_deepmind

from dqn.configs import default_config
from dqn.memory import initialize_memory, transition
from dqn.networks import ValueFunction


def get_wrapper_by_name(env, classname):
  currentenv = env
  while True:
    if classname in currentenv.__class__.__name__:
      return currentenv
    elif isinstance(env, gym.Wrapper):
      currentenv = currentenv.env
    else:
      raise ValueError("Couldn't find wrapper named %s" % classname)


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


def make_saver(sess, experiment_dir):
  checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
  checkpoint_path = os.path.join(checkpoint_dir, "model")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  saver = tf.train.Saver(max_to_keep=5)
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_checkpoint:
    print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)
  return saver, checkpoint_path


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
    episode_rewards = get_wrapper_by_name(
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
  
  env = gym.make('SpaceInvaders-v0')
  # env = gym.make('Pong-v0')
  experiment_dir = os.path.abspath("./experiments3/{}/{}".format(
    # str(datetime.now()).replace(' ', '_'),
    env.spec.id, 'dddqn' if _config.use_dddqn else 'dqn'))
  env = gym.wrappers.Monitor(env, experiment_dir + "/gym", force=True)
  env = wrap_deepmind(env, dim=_config.frame_dim)
  
  atari_actions = np.arange(env.action_space.n, dtype=np.int32)
  
  # Initialize networks.
  with tf.variable_scope('q_network'):
    q_network = ValueFunction(_config,
                              env.observation_space,
                              env.action_space,
                              summaries_dir=experiment_dir)
  with tf.variable_scope('target'):
    target = ValueFunction(_config, env.observation_space, env.action_space, q_network)
  # Epsilon
  eps = np.linspace(_config.epsilon_start, _config.epsilon_end, _config.epsilon_decay_steps)
  
  sess = make_session()
  evaluaor = Evaluate(sess, env, q_network)
  policy = EpsGreedy(q_network, env.action_space.n)
  initialize_variables(sess)
  saver, checkpoint_path = make_saver(sess, experiment_dir)
  
  # Initialize memory
  memory = initialize_memory(sess, env, _config, q_network)
  # Initialize policy
  total_step = sess.run(tf.train.get_global_step())
  last_timestep = 0
  print('total_step', total_step)
  
  for episode in range(_config.num_episodes):
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
      
      loss = q_network.update_step(sess, batch_transition.observ, batch_transition.action, target_values,
                                   total_step)
      print('\r({}/{}) loss: {}'.format(
        total_step, _config.max_total_step_size, loss), end='', flush=True)
      
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
    last_timestep = total_step
    saver.save(sess, checkpoint_path, global_step=tf.train.get_global_step())


if __name__ == '__main__':
  tf.app.run()