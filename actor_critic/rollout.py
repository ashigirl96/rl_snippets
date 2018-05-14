"""Rollout one episode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np

from reinforce.policy import Policy

Transition = collections.namedtuple('Transition',
                                    'observ, reward, done, action, next_observ, raw_return, return_')
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


def rollouts(env, policy: Policy):
    """
    Args:
        env: OpenAI Gym wrapped by agents.wrappers
        policy(Policy): instance of Policy
        reward_filter(MeanStdFilter): Use ray's MeanStdFilter for calculate easier

    Returns:
        1 episode(rollout) that is sequence of trajectory.
    """
    raw_return = 0
    return_ = 0
    observ = env.reset()
    _observ = observ[np.newaxis, ...]
    
    trajectory = []
    for t in itertools.count():
        # a_t ~ pi(a_t | s_t)
        action = policy.compute_action(_observ)
        
        next_observ, reward, done, _ = env.step(action)
        
        # Make trajectory sample.
        trajectory.append(Transition(observ, reward, done, action, next_observ, raw_return, return_))
        
        if done:
            break
        
        # s_{t+1} ← s_{t}
        observ = next_observ
        _observ = observ[np.newaxis, ...]
    return trajectory


def evaluate_policy(policy):
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


def video_evaluate_policy(policy, env_name):
    """
    Args:
        policy(Policy): instance of Policy
    Returns:
        score
    """
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, './video', force=True)
    
    raw_return = 0
    observ = env.reset()
    frames = []
    
    for t in itertools.count():
        # a_t ~ pi(a_t | s_t)
        frames.append(env.render(mode='rgb_array'))
        action = policy.compute_action(observ)
        observ, reward, done, _ = env.step(action)
        raw_return += reward
        
        if done:
            break
    return np.asarray(frames)