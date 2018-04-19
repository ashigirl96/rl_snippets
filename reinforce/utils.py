"""Utility function and class for implement RENFORCE"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def plot_agent_stats(results, window=5):
    policy_loss, val_loss, eval = [result for result in zip(*results)]
    df = pd.DataFrame({'policy_loss': policy_loss,
                 'value_func_loss': val_loss,
                 'eval': eval})
    df = df.rolling(window=window)
    df = df.mean()
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    # plt.xticks(df.index.values)
    plt.xlabel('Number of Episodes')
    plt.title('REINFORCE algorithm.')
    plt.xlim((0, df.index.values[-1]))
    plt.ylim((0, 200))
    axes[0].plot(df['policy_loss'], label='Policy Loss.')
    axes[0].set_ylabel('Policy Loss.')
    
    axes[1].plot(df['value_func_loss'], label='Value Function Loss')
    axes[1].set_ylabel('Value Function Loss.')
    
    axes[2].plot(df['eval'], label='Evaluation Value of Agent.')
    axes[2].set_ylabel('Evaluation Value of Agent.')
    
    



print('{0}{1}{2}'.format(bcolors.OKGREEN, get_available_gpus(), bcolors.ENDC))