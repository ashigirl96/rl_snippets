"""Utility function and class for implement RENFORCE"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym.spaces
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from agents.tools import AttrDict
from tensorflow.python.client import device_lib

import gym

sns.set(palette='husl')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def plot_agent_stats(results, title, window=5):
    policy_loss, val_loss, eval = [result for result in zip(*results)]
    df = pd.DataFrame({
        'policy_loss': policy_loss,
        'value_func_loss': val_loss,
        'eval': eval})
    df = df.rolling(window=window)
    df = df.mean()
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    # plt.xticks(df.index.values)
    plt.xlabel('Number of Episodes')
    plt.title(title)
    plt.xlim((0, df.index.values[-1]))
    plt.ylim((0, 500))
    axes[0].plot(df['policy_loss'], label='Policy Loss.')
    axes[0].set_ylabel('Policy Loss.')
    
    axes[1].plot(df['value_func_loss'], label='Value Function Loss')
    axes[1].set_ylabel('Value Function Loss.')
    
    axes[2].plot(df['eval'], label='Evaluation Value of Agent.')
    axes[2].set_ylabel('Evaluation Value of Agent.')


def get_env_info(env: gym.Env):
    is_continuous_action = isinstance(env.action_space, gym.spaces.Box) or False
    is_continuous_observ = isinstance(env.observation_space, gym.spaces.Box) or False
    
    if is_continuous_action:
        action_shape = (None, *env.action_space.shape)
        if len(action_shape) != 2:
            raise NotImplementedError('Never implement continuous multi-actions')
        action_n = action_shape[1]
    else:
        action_shape = ()
        action_n = env.action_space.n
    
    if is_continuous_observ:
        observ_shape = (None, *env.observation_space.shape)
    else:
        raise NotImplementedError('Never implement discrete observation')
    
    return locals()


print('{0}{1}{2}'.format(bcolors.OKGREEN, get_available_gpus(), bcolors.ENDC))