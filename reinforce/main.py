"""main file coded the reinforce algorithm.
I will replace this code to reusability."""
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from agents.tools import AttrDict

from reinforce.agent import REINFORCE
from reinforce.rollout import evaluate_policy
from reinforce.utils import plot_agent_stats


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
    num_episodes = 100
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