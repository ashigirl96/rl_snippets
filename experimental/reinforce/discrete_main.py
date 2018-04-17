"""One file REINFORCE algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np
import tensorflow as tf
from agents.tools.attr_dict import AttrDict
from agents.tools.wrappers import ConvertTo32Bit
from ray.experimental.tfutils import TensorFlowVariables
import matplotlib.pyplot as plt
from reinforce.utils import plot_agent_stats

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
        optimizer = tf.train.AdamOptimizer(0.01)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        # self.train_op = optimizer.minimize(self.loss)
        
        self.variables = TensorFlowVariables(self.loss, self.sess)
    
    def _build_model(self):
        """Build TensorFlow policy model."""
        # To look clearly, whether I use bias.
        use_bias = self.config.use_bias
        
        # Placeholders. observ = Box(4,), action = Discrete(2,)
        # I have to describe which action do I select.
        self.observ = tf.placeholder(tf.float32, (None, 4), name='observ')
        self.action = tf.placeholder(tf.int32, name='action')
        self.expected_value = tf.placeholder(tf.float32, name='expected_value')
        x = tf.layers.dense(self.observ, 2, use_bias=use_bias,
                            kernel_initializer=tf.zeros_initializer,
                            activation=None)
        self.logits = x
        self.action_probs = tf.squeeze(tf.nn.softmax(self.logits))
    
    def _set_loss(self):
        # For use `tf.nn.sparse_softmax_cross_entropy_with_logits`,
        # The shape of action should be `(1, ...)`
        picked_action_prob = tf.gather(self.action_probs, self.action)
        log_prob = -tf.log(picked_action_prob) * self.expected_value
        log_prob = tf.check_numerics(log_prob, 'log_prob')
        self.loss = log_prob
    
    def compute_action(self, observ):
        """Generate action from \pi(a_t | s_t) that is neural network.
        
        Args:
            observ: Observation generated by gym.Env.observation.

        Returns:
            Action
        """
        assert observ.shape == (1, 4)
        action_probs = self.sess.run(self.action_probs, feed_dict={self.observ: observ})
        
        # Note selection disappeared only discrete action model.
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs).astype(np.int64)
        assert isinstance(action, np.int64)
        return action
    
    def apply(self, observ, action, expected_value):
        """Apply the gradients to weights.
        Args:
            observ: Observation of Gym.
            action: Action of agent.
            expected_value:  expected value (i.g. advantage)

        Returns:
            loss: loss of train operation.
        """
        _, loss = self.sess.run(
            [self.train_op, self.loss], feed_dict={
                self.observ: observ,
                self.action: action,
                # self.policy.expected_value: transition.return_,
                self.expected_value: expected_value,
            })
        return loss


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
        optimizer = tf.train.AdamOptimizer(0.01)
        self.train_op = optimizer.minimize(self.loss)
        
        self.variables = TensorFlowVariables(self.loss, self.sess)
    
    def _build_model(self):
        # To look clearly, whether I use bias.
        use_bias = self.config.use_bias
        
        self.observ = tf.placeholder(tf.float32, (None, 4), name='observ')
        self.return_ = tf.placeholder(tf.float32, name='return_')
        x = tf.layers.dense(self.observ, 1, use_bias=use_bias,
                            kernel_initializer=tf.zeros_initializer)
        self.logits = x
    
    def _set_loss(self):
        losses = tf.losses.mean_squared_error(labels=self.return_,
                                              predictions=self.logits)
        self.loss = tf.reduce_mean(losses)
    
    def predict(self, observ):
        baseline = self.sess.run([self.logits],
                                 feed_dict={self.observ: observ})
        return baseline
    
    def apply(self, observ, return_):
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.observ: observ, self.return_: return_})
        return loss


class REINFORCE(object):
    
    def __init__(self, config):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        env = gym.make(config.env_name)
        self.config = config
        self.env = ConvertTo32Bit(env)
        self.policy = Policy(sess=self.sess, config=config)
        self.value_func = ValueFunction(sess=self.sess, config=config)
        
        self._init()
    
    def _init(self):
        self.sess.run(tf.global_variables_initializer())
    
    def compute_trajectory(self):
        trajectory = rollouts(self.env,
                              self.policy)
        return trajectory
    
    def _train(self):
        """REINFORCE Algorithm.

        Returns: Losses each timestep.
        """
        trajectory = self.compute_trajectory()
        
        policy_losses = []
        value_func_losses = []
        for t, transition in enumerate(trajectory):
            baseline = self.value_func.predict(transition.observ)
            return_ = sum(1. ** i * t.reward for i, t in enumerate(trajectory[t:]))
            advantage = return_ - baseline
            loss_ = self.value_func.apply(transition.observ, return_)
            loss = self.policy.apply(transition.observ, transition.action, advantage)
            policy_losses.append(loss)
            value_func_losses.append(loss_)
        return policy_losses, value_func_losses
    
    def train(self, num_episodes):
        saver = tf.train.Saver()
        for _ in range(num_episodes):
            losses = self._train()
            yield losses
        saver.save(self.sess, './reinforce_debug')


def rollouts(env, policy: Policy):
    """
    Args:
        env: OpenAI Gym wrapped by agents.wrappers
        policy(Policy): instance of Policy

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
        next_observ = next_observ[np.newaxis, ...]
        
        # Make trajectory sample.
        trajectory.append(Transition(observ, reward, done, action, next_observ, raw_return, return_))
        
        if done:
            break
        
        # s_{t+1} ← s_{t}
        observ = next_observ
    return trajectory


def evaluate_policy(policy, config):
    """
    Args:
        policy(Policy): instance of Policy
    Returns:
        score
    """
    # env = gym.make(config.env_name)
    env = gym.make('CartPole-v0')
    
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


def default_config():
    # Whether use bias on layer
    use_bias = True
    # OpenAI Gym environment name
    env_name = 'CartPole-v0'
    # Discount Factor (gamma)
    discount_factor = 1.
    # Learning rate
    learning_rate = 0.1
    # Number of episodes
    num_episodes = 200
    # Activation function used in dense layer
    activation = tf.nn.relu
    # Epsilon-Greedy Policy
    eps = 0.1
    
    return locals()


def main(_):
    config = AttrDict(default_config())
    # Define Agent that train with REINFORCE algorithm.
    agent = REINFORCE(config)
    
    mean_policy_losses = []
    mean_valfunc_losses = []
    mean_evals = []

    # Train for num_iters times.
    for i, (policy_loss, val_func_loss) in enumerate(agent.train(config.num_episodes)):
        mean_policy_losses.append(np.mean(policy_loss))
        mean_valfunc_losses.append(np.mean(val_func_loss))
        mean_evals.append(np.mean([evaluate_policy(agent.policy, config) for _ in range(5)]))
    
        print('\rEpisode {}/{} policy loss ({}), value loss ({}), eval ({})'.format(
            i, config.num_episodes,
            mean_policy_losses[-1], mean_valfunc_losses[-1], mean_evals[-1]),
            end='', flush=True)

    stats = [mean_policy_losses, mean_valfunc_losses, mean_evals]
    plot_agent_stats(stats)
    plt.show()


if __name__ == '__main__':
    tf.set_random_seed(42)
    np.random.seed(42)
    tf.app.run()