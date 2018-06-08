"""DQN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gym
import tensorflow as tf
from agents import tools

from dqn import networks
from dqn.configs import default_config
from dqn.memory import initialize_memory


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


def main(_):
    env = gym.make('SpaceInvaders-v0')
    
    _config = tools.AttrDict(default_config())
    
    # Initialize networks.
    _structure = functools.partial(_config.network,
                                   _config,
                                   env.observation_space,
                                   env.action_space)
    _target = tf.make_template('target', _structure)
    _network = tf.make_template('network', _structure)
    network = _network()
    target = _target(network.picked_action)
    
    predicted_target = networks.value_prediction(target)
    loss = networks.compute_loss(network, predicted_target.value)
    with tf.device('/gpu:0'):
        optimizer = tf.train.RMSPropOptimizer(0.00025)
        train_op = optimizer.minimize(loss)
    
    sess = make_session()
    initialize_variables(sess)
    
    # Initialize replay buffer.
    buffer = initialize_memory(sess, env, _config)
    
    for _ in range(2):
        batch_transition = buffer.sample(_config.batch_size)
        
        feed_dict = dict()
        feed_dict.update(predicted_target.feed(batch_transition.reward,
                                               batch_transition.terminal))
        feed_dict.update(network.feed(batch_transition.observ,
                                      batch_transition.action))
        feed_dict.update(target.feed(batch_transition.next_observ))
        _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
        print('loss {}'.format(loss_))


if __name__ == '__main__':
    tf.app.run()