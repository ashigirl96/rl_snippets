"""Test network"""

import functools

import gym
import tensorflow as tf
from agents import tools

from dqn.configs import default_config


class NetworkTest(tf.test.TestCase):
    def test_initialize(self):
        env = gym.make('SpaceInvaders-v0')
        
        _config = tools.AttrDict(default_config())
        
        _structure = functools.partial(_config.network,
                                       _config,
                                       env.observation_space,
                                       env.action_space)
        _network = tf.make_template('network', _structure)
        _target = tf.make_template('target', _structure)
        network = _network()
        target = _target(network.picked_action)
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
    
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     init.run()
    #     buffer = initialize_memory(sess, env, _config)
    #     batch_transition = buffer.sample(_config.batch_size)
    #     network_value = sess.run(
    #         network.value, feed_dict=network.feed(batch_transition.observ, batch_transition.action))
    #     target_value = sess.run(
    #         target.value, feed_dict=target.feed(batch_transition.next_observ, batch_transition.action))
    #     assert network_value.shape == (_config.batch_size, env.action_space.n)
    #     assert target_value.shape == (_config.batch_size, env.action_space.n)


if __name__ == '__main__':
    tf.test.main()