"""Construct Q-network and target network"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
from agents import tools

from dqn import utility


class ValueFunction(object):
    def __init__(self, config,
                 observ_space: gym.Env.observation_space,
                 action_space: gym.Env.action_space,
                 q_network=None,
                 processed_observ_shape=(84, 84, 4)):
        if not isinstance(observ_space, gym.spaces.Box):
            raise ValueError('Unexpected observation space.')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Unexpected action space.')
        self._config = config
        self._observ_space = observ_space
        self._action_space = action_space
        self._processed_observ_shape = processed_observ_shape
        
        # Build value function.
        self._value_network(processed_observ_shape)
        # Network variables.
        self.vars_ = utility.trainable_variables(self)
        # This process will be done by target network.
        if q_network:
            self._assign_op = self._assign(q_network)
            self._gamma = self._config.gamma
            self._value_prediction()
        # Loss Function.
        self._value_loss()
        
        # Construct optimizer.
        use_gpu = self._config.use_gpu
        with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
            optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=tf.train.get_or_create_global_step())
    
    def predict(self, sess: tf.Session, observ):
        return sess.run(self.value, feed_dict={self._observ: observ})
    
    def update_step(self, sess: tf.Session, observs, actions, target_values):
        feed_dict = {
            self._observ: observs,
            self._action: actions,
            self._target_value: target_values
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
    
    def assign(self, sess: tf.Session):
        sess.run(self._assign_op)
    
    def _value_network(self, processed_observ_shape):
        self._observ = tf.placeholder(tf.float32, [None, *processed_observ_shape], name='observ')
        self._action = tf.placeholder(tf.int32, [None], name='action')  # [2, 1, 1, 0] if batch_size is 4
        action_size = self._action_space.n
        
        with tf.variable_scope('value'):
            x = self._observ
            for size in self._config.conv_value_layers:
                x = tf.layers.conv2d(x, filters=size[0], kernel_size=size[1], strides=size[2], activation=tf.nn.relu)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 512, tf.nn.relu)
            value = tf.layers.dense(x, action_size)
            self.value = tf.check_numerics(value, value.name)
            
            # Q-network use this operation.
            self.picked_action = tf.argmax(self.value, axis=1)
            # Look at experimental/gather.py
            gather_indices = tf.range(self._config.batch_size) * action_size + self._action
            self.picked_value = tf.gather(tf.reshape(self.value, [-1]), gather_indices)
    
    def _value_loss(self):
        """Loss function for vanilla DQN and DDQN.
        
        Args:
            network: Trainable network.
            scope: Scope name.

        Returns:
            loss:
        """
        self._target_value = tf.placeholder(tf.float32, [None])
        losses = tf.losses.mean_squared_error(
            self._target_value,
            self.picked_value)
        self.loss = tf.reduce_mean(losses)
    
    def _value_prediction(self):
        """Operation for predict target value.
        
        This function used only target network.
        """
        self._reward = tf.placeholder(tf.float32, [None], name='reward')
        self._terminal = tf.placeholder(tf.bool, [None], name='terminal')
        not_terminal = tf.cast(tf.logical_not(self._terminal), tf.float32)
        self.y = self._reward + self._gamma * tf.multiply(not_terminal, self.picked_value)
    
    def best_action(self, sess: tf.Session, next_observs):
        feed_dict = {self._observ: next_observs}
        return sess.run(self.picked_action, feed_dict=feed_dict)
    
    def estimate(self, sess: tf.Session, rewards, terminals, observs, actions):
        """Estimate target value
        
        Args:
            sess: tf.Session
            rewards: batch of rewards
            terminals: batch of terminals(done)
            actions: batch of actions
        """
        feed_dict = {
            self._reward: rewards,
            self._terminal: terminals,
            # For calculate next value
            self._observ: observs,
            self._action: actions,
        }
        return sess.run(self.y, feed_dict=feed_dict)
    
    def _assign(self, other_network):
        """Assign Q-network's weights to target networks.
        
        Args:
            other_network: Q-network.
        """
        with tf.name_scope('update_target'):
            update_target = tools.nested.map(lambda target_var, network_var:
                                             target_var.assign(network_var),
                                             self.vars_, other_network.vars_)
            return tf.group(*update_target)