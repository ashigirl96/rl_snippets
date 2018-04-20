"""main file coded the reinforce algorithm.
I will replace this code to reusability."""
from __future__ import division
from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import ray
import tensorflow as tf
from agents.tools import AttrDict

from reinforce.agent import REINFORCE
from reinforce.policy import RemotePolicy
from reinforce.utils import plot_agent_stats


def default_config():
    # Whether use bias on layer
    use_bias = True
    # OpenAI Gym environment name
    env_name = 'CartPole-v1'
    # Discount Factor (gamma)
    discount_factor = .995
    # Learning rate
    learning_rate = 0.1
    # Number of episodes
    num_episodes = 100
    # Activation function used in dense layer
    activation = tf.nn.relu
    # Epsilon-Greedy Policy
    eps = 0.1
    
    # Use GPUS
    use_gpu = True
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    
    return locals()


def main(_):
    
    config = AttrDict(default_config())
    # Define Agent that train with REINFORCE algorithm.
    agent = REINFORCE(config)
    
    remote_policies = [RemotePolicy.remote(config) for _ in range(5)]
    
    train_results = []

    start_time = time.time()
    # Train for num_iters times.
    for i, result in enumerate(agent.train(config.num_episodes)):
        evals = agent.evaluate(remote_policies)
        result = result._replace(eval=np.mean(evals))
        
        print('\rEpisode {}/{} policy loss ({}), value loss ({}), eval ({})'.format(
            i, config.num_episodes,
            result.policy_loss, result.val_loss, result.eval),
            end='', flush=True)
        
        train_results.append(result)
    
    end_time = time.time()
    duration = end_time - start_time
    # 100 episodes, Process duration: 40.74085235595703[s] using GPU
    print('\nProcess duration: {0}[s]'.format(duration))
    
    plot_agent_stats(train_results)
    plt.show()


if __name__ == '__main__':
    ray.init(num_cpus=5, redirect_output=True)
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.set_random_seed(42)
    np.random.seed(42)
    tf.app.run()