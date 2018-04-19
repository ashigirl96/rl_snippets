"""main file coded the reinforce algorithm.
I will replace this code to reusability."""
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ray
import tensorflow as tf
from agents.tools import AttrDict

from reinforce.agent import REINFORCE
from reinforce.policy import RemotePolicy
from reinforce.rollout import evaluate_policy
from reinforce.utils import plot_agent_stats


def default_config():
    # Whether use bias on layer
    use_bias = True
    # OpenAI Gym environment name
    env_name = 'CartPole-v0'
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
    remote_evaluate_policy = ray.remote(evaluate_policy)
    import time
    start_time = time.time()
    
    config = AttrDict(default_config())
    # Define Agent that train with REINFORCE algorithm.
    agent = REINFORCE(config)
    
    remote_policies = [RemotePolicy.remote(config) for _ in range(5)]
    
    mean_policy_losses = []
    mean_valfunc_losses = []
    mean_evals = []
    
    # Train for num_iters times.
    for i, (policy_loss, val_func_loss) in enumerate(agent.train(config.num_episodes)):
        mean_policy_losses.append(np.mean(policy_loss))
        mean_valfunc_losses.append(np.mean(val_func_loss))
        weights = agent.policy.get_weights()
        weights_id = ray.put(weights)
        
        evaluate_ids = [remote_policies[i].evaluate.remote(weights_id) for i in range(5)]
        evals = ray.get(evaluate_ids)
        mean_evals.append(np.mean(evals))

        print('\rEpisode {}/{} policy loss ({}), value loss ({}), eval ({})'.format(
            i, config.num_episodes,
            mean_policy_losses[-1], mean_valfunc_losses[-1], mean_evals[-1]),
            end='', flush=True)

    stats = [mean_policy_losses, mean_valfunc_losses, mean_evals]

    end_time = time.time()
    duration = end_time - start_time
    print('\nProcess duration: {0}[s]'.format(duration))

    plot_agent_stats(stats)
    plt.show()


if __name__ == '__main__':
    ray.init(num_cpus=5, redirect_output=True)
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.set_random_seed(42)
    np.random.seed(42)
    tf.app.run()