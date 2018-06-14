"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import gym
import numpy as np
import tensorflow as tf
from agents import tools

from dqn.configs import default_config
from dqn.memory import initialize_memory, transition
from dqn.networks import ValueFunction
# from dqn.preprocess import atari_preprocess
from ray.rllib.utils.atari_wrappers import wrap_deepmind


def make_session(num_cpu=8):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


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


def make_saver(sess):
    checkpoint_dir = './logdir'
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(max_to_keep=5)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    summary_writer = tf.summary.FileWriter('./logdir', sess.graph)
    return saver, checkpoint_path


def main(_):
    env = gym.make('SpaceInvaders-v0')
    env = wrap_deepmind(env)
    
    atari_actions = np.arange(env.action_space.n, dtype=np.int32)
    
    _config = tools.AttrDict(default_config())
    
    # Initialize networks.
    with tf.variable_scope('q_network'):
        q_network = ValueFunction(_config, env.observation_space, env.action_space)
    with tf.variable_scope('target'):
        target = ValueFunction(_config, env.observation_space, env.action_space, q_network)
    # Initialize global step
    # Epsilon
    eps = np.linspace(_config.epsilon_start, _config.epsilon_end, _config.epsilon_decay_steps)
    
    sess = make_session()
    initialize_variables(sess)
    saver, checkpoint_path = make_saver(sess)
    
    # Initialize memory
    memory = initialize_memory(sess, env, _config)
    # Initialize policy
    policy = eps_greedy_policy(q_network, env.action_space.n)
    
    total_step = sess.run(tf.train.get_global_step())
    print('total_step', total_step)
    
    for episode in range(_config.num_episodes):
        observ = env.reset()
        observ = atari_preprocess(sess, observ)
        observ = np.stack([observ] * 4, axis=2)
        for t in itertools.count():
            action_prob = policy(sess, observ,
                                 eps[min(total_step, _config.epsilon_decay_steps - 1)])
            action = np.random.choice(atari_actions, size=1, p=action_prob)[0]
            next_observ, reward, terminal, _ = env.step(action)
            # next_observ = atari_preprocess(sess, next_observ)
            next_observ = np.concatenate(
                [observ[..., 1:], next_observ[..., None]], axis=2)
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
            
            if total_step % _config.update_target_estimator_every == 0:
                print('\nUpdate Target Network...')
                target.assign(sess)
            
            if terminal:
                break
            
            total_step += 1
        saver.save(sess, checkpoint_path, global_step=tf.train.get_global_step())


if __name__ == '__main__':
    tf.app.run()