"""One file REINFORCE algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from agents.tools.attr_dict import AttrDict
from agents.tools.wrappers import ConvertTo32Bit
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.utils.filter import MeanStdFilter

from reinforce.utils import bcolors

Transition = collections.namedtuple('Transition',
                                    'observ, reward, done, action, next_observ, raw_return, return_')
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


class Policy(object):
    
    def __init__(self, sess: tf.Session, config):
        """Neural network policy that compute action given observation.
        
        Args:
            config: Useful configuration, almost use as const.
        """
        self.sess = sess
        
        self.config = config
        self._build_model()
        self._set_loss()
        optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        
        self.variables = TensorFlowVariables(self.loss, self.sess)
        self.sess.run(tf.global_variables_initializer())
    
    def _build_model(self):
        """Build TensorFlow policy model."""
        self.observ = tf.placeholder(tf.float32, (None, 3), name='observ')
        self.action = tf.placeholder(tf.float32, (None), name='action')
        self.expected_value = tf.placeholder(tf.float32, name='expected_value')
        x = tf.layers.dense(self.observ, 100, use_bias=False)
        x = tf.layers.dense(x, 100, use_bias=False)
        x = tf.layers.dense(x, 1, use_bias=False)
        x = tf.clip_by_value(x, -2., 2.)
        self.model = x
    
    def _set_loss(self):
        # TODO: How to implement loss function.
        prob = tf.nn.softmax(self.model)
        action_prob = self.action
        losses = tf.losses.mean_squared_error(prob, action_prob)
        log_prob = -tf.log(losses) * tf.stop_gradient(self.expected_value)
        log_prob = tf.check_numerics(log_prob, 'log_prob')
        self.loss = log_prob
    
    def compute_action(self, observ):
        """Generate action from \pi(a_t | s_t) that is neural network.
        
        Args:
            observ: Observation generated by gym.Env.observation.

        Returns:
            (Lights, Camera) Action
        """
        assert observ.shape == (1, 3)
        action = self.sess.run(self.model, feed_dict={self.observ: observ})
        return action[0]


# TODO: I WILL IMPLEMENT.
class ValueFunction(object):
    
    def __init__(self, sess: tf.Session, config):
        """Neural network policy that compute action given observation.
        
        Args:
            config: Useful configuration, almost use as const.
        """
        self.sess = sess
        
        self.config = config
        self._build_model()
        self._set_loss()
        optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        
        self.variables = TensorFlowVariables(self.loss, self.sess)
        self.sess.run(tf.global_variables_initializer())
    
    def _build_model(self):
        pass


class REINFORCE(object):
    
    def __init__(self, config):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        env = gym.make(config.env_name)
        self.config = config
        self.env = ConvertTo32Bit(env)
        self.policy = Policy(sess=self.sess, config=config)
        
        self._init()
    
    def _init(self):
        self.reward_filter = MeanStdFilter((), clip=5.)
    
    def compute_trajectory(self):
        trajectory = rollouts(self.env,
                              self.policy,
                              self.reward_filter,
                              self.config)
        return trajectory
    
    def _train(self):
        """REINFORCE Algorithm.
        
        Returns: Losses each timestep.
        """
        trajectories = []
        for _ in range(self.config.num_episodes):
            trajectory = self.compute_trajectory()
            trajectories.append(trajectory)
        
        losses = []
        for trajectory in trajectories:
            for transition in trajectory:
                _, loss = self.sess.run(
                    [self.policy.train_op, self.policy.loss], feed_dict={
                        self.policy.observ: transition.observ,
                        self.policy.action: transition.action,
                        self.policy.expected_value: transition.return_})
                losses.append(loss)
        return losses
    
    def train(self, num_iters):
        for i in range(num_iters):
            losses = self._train()
            yield losses


def rollouts(env, policy: Policy, reward_filter: MeanStdFilter, config):
    """
    Args:
        env: OpenAI Gym wrapped by agents.wrappers
        policy(Policy): instance of Policy
        reward_filter(MeanStdFilter): Use ray's MeanStdFilter for calculate easier
        config: Useful configuration, almost use as const.

    Returns:
        1 episode(rollout) that is sequence of trajectory.
    """
    raw_return = 0
    return_ = 0
    observ = env.reset()
    observ = observ[np.newaxis, ...]
    
    trajectory = []
    for t in itertools.count():
        # a_t ~ pi(a_t | s_t)
        action = policy.compute_action(observ)
        
        next_observ, reward, done, _ = env.step(action)
        
        # This rollout does not provide batch observ and action.
        # it reshape (2,) → (1, 2), (1,) → (1, 1) for TensorFlow placeholder.
        next_observ = next_observ[np.newaxis, ...]
        action = action[np.newaxis, ...]
        
        # Adjust reward
        reward = reward_filter(reward)
        raw_return += reward
        return_ += reward * config.discount_factor ** t
        
        # Make trajectory sample.
        trajectory.append(Transition(observ, reward, done, action, next_observ, raw_return, return_))
        
        # s_{t+1} ← s_{t}
        observ = next_observ
        
        if done:
            break
    return trajectory


def default_config():
    # Whether use bias on layer
    use_bias = False
    # OpenAI Gym environment name
    # env_name = 'MountainCarContinuous-v0'
    env_name = 'Pendulum-v0'
    # Discount Factor (gamma)
    discount_factor = 0.995
    # Learning rate
    learning_rate = 1e-5
    # Number of episodes
    num_episodes = 1
    
    return locals()


def evaluate_policy(policy, config):
    """
    Args:
        policy(Policy): instance of Policy
    Returns:
        score
    """
    env = gym.make(config.env_name)
    
    raw_return = 0
    observ = env.reset()
    observ = observ[np.newaxis, ...]
    
    for t in itertools.count():
        # a_t ~ pi(a_t | s_t)
        action = policy.compute_action(observ)
        observ, reward, done, _ = env.step(action)
        observ = observ[np.newaxis, ...]
        raw_return += reward
        
        if done:
            break
    return raw_return


def main(_):
    config = AttrDict(default_config())
    # Define Agent that train with REINFORCE algorithm.
    agent = REINFORCE(config)
    
    # Train for num_iters times.
    episode_loss = []
    for i, losses in enumerate(agent.train(num_iters=100)):
        loss = np.mean(losses)
        message = 'episode: {0}, loss: {1}'.format(i, loss)
        print('{0}{1}{2}'.format(bcolors.HEADER, message, bcolors.ENDC))
        episode_loss.append(loss)
    x = np.arange(len(episode_loss))
    plt.plot(x, episode_loss)
    plt.show()


if __name__ == '__main__':
    tf.app.run()