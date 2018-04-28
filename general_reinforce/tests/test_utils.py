import gym
from general_reinforce.utils import get_env_info


def test_discrete_get_env_info():
    env = gym.make('CartPole-v1')
    env_info = get_env_info(env)
    
    assert env_info.is_continuous_action is False
    assert env_info.is_continuous_observ is True
    assert env_info.action_n == 2
    assert env_info.observ_shape == (None, 4)
    assert env_info.action_shape == ()


def test_continuous_get_env_info():
    env = gym.make('BipedalWalker-v2')
    env_info = get_env_info(env)
    
    assert env_info.is_continuous_action is True
    assert env_info.is_continuous_observ is True
    assert env_info.action_n == 4
    assert env_info.observ_shape == (None, 24)
    assert env_info.action_shape == (None, 4)