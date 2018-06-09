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
        self._config = config
        self._observ_space = observ_space
        self._action_space = action_space
        self._processed_observ_shape = processed_observ_shape
        
        # Build value function.
        self._value_network(processed_observ_shape)
        # Network variables.
        self.vars_ = utility.trainable_variables(self)
        if q_network:
            self._assign_op = self._assign(q_network)
        # Loss Function.
        self._value_loss()
        
        
        # Construct optimizer.
        use_gpu = self._config.use_gpu
        learning_rate = self._config.learning_rate
        with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)
    
    def predict(self, sess: tf.Session, observ):
        return sess.run(self.value, feed_dict={self._observ: observ})
    
    def update_step(self, sess: tf.Session, batch_transition):
        pass
    
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


# Q-Network and Target Network.
def feed_forward_value(config,
                       observ_space: gym.Env.observation_space,
                       action_space: gym.Env.action_space,
                       action=None,
                       processed_observ_shape=(84, 84, 4)):
    """Independent feed forward networks for policy and value.
    
    The policy network outputs the mean action and the log standard deviation
    is learned as independent parameter vector.
    Args:
      config: Configuration object.
      observ_space: Sequences of observations.
      action_space: Action space of the environment.
      action: For target networks which have to use network's action.
      processed_observ_shape: Shape of observation which processed.
    Raises:
      ValueError: Unexpected action space.
    Returns:
      Attribute dictionary containing the policy, value, and unused state.
    """
    eps = 0.2
    
    if not isinstance(observ_space, gym.spaces.Box):
        raise ValueError('Unexpected observation space.')
    if not isinstance(action_space, gym.spaces.Discrete):
        raise ValueError('Unexpected action space.')
    action_size = action_space.n
    temporary_action = tf.Variable(
        tf.zeros([config.batch_size * action_size]), False)
    observ = tf.placeholder(tf.float32, [None, *processed_observ_shape], name='observ')
    
    training = tf.placeholder(tf.bool, name='training')
    action = tf.placeholder(tf.int32, [None], name='action')  # [2, 1, 1, 0] if batch_size is 4
    if isinstance(action, tf.Tensor):
        # Target
        feed = lambda observ_, training_: {observ: observ_, training: training_}
    else:
        # Network
        def feed(observ_, action_=None, training_=True):
            if action is None:
                return {observ: observ_, training: training_}
            return {observ: observ_, action: action_, training: training_}
    
    with tf.variable_scope('value'):
        x = observ
        for size in config.conv_value_layers:
            x = tf.layers.conv2d(x, filters=size[0], kernel_size=size[1], strides=size[2], activation=tf.nn.relu)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 512, tf.nn.relu)
        value = tf.layers.dense(x, action_size)
        value = tf.check_numerics(value, value.name)
        
        # Look at experimental/gather.py
        gather_indices = tf.range(config.batch_size) * action_size + action
        picked_value = tf.gather(tf.reshape(value, [-1]), gather_indices)
    
    with tf.variable_scope('policy'):
        _action = tf.argmax(value, axis=1)
        _action = tf.cast(_action, tf.int32)
        
        # TODO: have to test it.
        def _eps_greedy():
            eps_m = eps / action_size
            ref = temporary_action.assign(tf.ones([config.batch_size * action_size]) * eps_m)
            indices = tf.range(config.batch_size) * action_size + _action
            updates = tf.ones([config.batch_size]) * eps_m + 1 - eps
            update = tf.scatter_nd_update(ref, indices[:, None], updates)
            update = tf.reshape(tf.convert_to_tensor(update), [config.batch_size, action_size])
            update = tf.log(update)
            picked_action = tf.cast(tf.multinomial(update, 1), tf.int32)
            return picked_action
        
        def _greedy():
            return _action
        
        picked_action = tf.cond(training, _eps_greedy, _greedy)
    
    return tools.AttrDict(
        value=value, picked_value=picked_value, picked_action=picked_action, feed=feed)


def value_prediction(target, scope='compute_target'):
    """Operation for predict target value.
    
    Args:
        target: Target Network
        scope: Scope name
    """
    feed = lambda reward_, terminal_: {reward: reward_, terminal: terminal_}
    with tf.name_scope(scope):
        reward = tf.placeholder(tf.float32, [None], name='reward')
        terminal = tf.placeholder(tf.bool, [None], name='terminal')
        not_terminal = tf.cast(tf.logical_not(terminal), tf.float32)
        value = reward + tf.multiply(not_terminal, target.picked_value)
        return tools.AttrDict(value=value, feed=feed)


def compute_loss(network, target_value, scope='compute_loss') -> tf.Tensor:
    """Loss function for vanilla DQN and DDQN.
    
    Args:
        network: Trainable network.
        target_value: Predicted target network value.
        scope: Scope name.

    Returns:
        loss:
    """
    with tf.name_scope(scope):
        losses = tf.losses.mean_squared_error(
            tf.stop_gradient(target_value),
            network.picked_value)
        loss = tf.reduce_mean(losses)
        return loss


def compute_action(sess, network, observ, training=True):
    return sess.run(network.picked_action, network.feed(observ[None, ...], True))