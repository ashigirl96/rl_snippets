"""Rollout return one trajectory"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import collections
import gym
import numpy as np

transition = collections.namedtuple(
    'transition', 'observ, action, next_observ, reward, terminal')


def rollout(policy, env: gym.Env):
    """Return Normalized trajectry.
    
    Args:
        policy: Policy that returns probability.
        env: OpenAI Gym.

    Returns:
        One trajectory.
    """
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError('This rollout is called only continuous ones.')
    if len(env.action_space.shape) > 1:
        raise NotImplementedError('Multi action cannot impemented.')
    
    observ = env.reset()
    trajectory = []
    
    for t in itertools.count():
        action_prob = policy.predict(observ)
        action = action_prob.sample()
        assert action.shape == env.action_space.shape
        next_observ, reward, done, _ = env.step(action)
        trajectory.append(transition(observ, action, next_observ, reward, done))
        if done:
            break
    
    # Normalize observations.
    observs, actions, next_observs, _, _ = map(np.asarray, zip(*trajectory))
    normalize_observ = np.stack([observs, next_observs], axis=0).mean()
    normalize_action = actions.mean()
    for i, t in enumerate(trajectory):
        t = t._replace(observ=(t.observ / normalize_observ),
                       action=(t.action / normalize_action),
                       next_observ=(t.next_observ / normalize_observ),
                       )
        trajectory[i] = t
    
    return trajectory