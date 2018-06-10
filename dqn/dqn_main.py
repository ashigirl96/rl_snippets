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


def eps_greedy_policy(q_network, action_size):
    def policy_fn(sess: tf.Session, observ, eps):
        eps_m = eps / action_size
        action_prob = np.ones(action_size) * eps_m
        q_value = q_network.predict(sess, observ[None, ...])
        best_action = np.argmax(q_value)
        action_prob[best_action] += 1 - eps
        return action_prob
    
    return policy_fn


def make_saver(sess, experiment_dir):
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # checkpoint_dir = './logdir'
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
        self._test_size = 3
        with tf.variable_scope('evaluate_return'):
            self._return = tf.Variable(tf.zeros(self._test_size),
                                       False, name='return')
            self._summary = tf.summary.merge([
                tf.summary.scalar('max_return', tf.reduce_max(self._return)),
                tf.summary.scalar('mean_return', tf.reduce_mean(self._return)),
                tf.summary.scalar('min_return', tf.reduce_min(self._return)),
            ])
    
    def evaluate(self):
        self._return.assign(tf.zeros_like(self._return))
        returns = []
        for i in range(self._test_size):
            returns.append(self._evaluate())
        _, summary, global_step = self._sess.run([
            self._return.assign(returns),
            self._summary,
            tf.train.get_global_step(),
        ])
        self._q_network._summary_writer.add_summary(
            summary, global_step=global_step)
    
    def _evaluate(self):
        """Evaluate one episodes and return score.
        
        Returns:
            score
        """
        return_ = 0.
        observ = self._env.reset()[None, ...]
        for t in itertools.count():
            # a_t ~ pi(a_t | s_t)
            assert observ.shape == (1, 80, 80, 4)
            action = self._q_network.best_action(self._sess, observ)
            observ, reward, done, _ = self._env.step(action)
            observ = observ[None, ...]
            return_ += reward
            if done:
                return return_


def main(_):
    env = gym.make('SpaceInvaders-v0')
    # env = gym.make('Pong-v0')
    env = wrap_deepmind(env)
    _config = default_config()
    
    experiment_dir = os.path.abspath("./experiments{}/{}".format(
        '_dddqn' if _config.use_dddqn else '_dqn', env.spec.id))
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
    initialize_variables(sess)
    saver, checkpoint_path = make_saver(sess, experiment_dir)
    
    # Initialize memory
    memory = initialize_memory(sess, env, _config, q_network)
    # Initialize policy
    policy = eps_greedy_policy(q_network, env.action_space.n)
    # rollout function
    evaluaor = Evaluate(sess, env, q_network)
    
    total_step = sess.run(tf.train.get_global_step())
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
            
            loss = q_network.update_step(sess, batch_transition.observ, batch_transition.action, target_values)
            print('\r({}/{}) loss: {}'.format(total_step, _config.max_total_step_size, loss), end='', flush=True)
            
            if total_step % _config.evaluate_every == 0:
                evaluaor.evaluate()
            
            if total_step % _config.update_target_estimator_every == 0:
                print('\nUpdate Target Network...')
                target.assign(sess)
            
            if terminal:
                break
            
            observ = next_observ
            total_step += 1
        saver.save(sess, checkpoint_path, global_step=tf.train.get_global_step())


if __name__ == '__main__':
    tf.app.run()