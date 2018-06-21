from os import path as osp

import gym
import tensorflow as tf
from gym import wrappers
from ray.rllib.utils.atari_wrappers import wrap_deepmind


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


def main(_):
    env = gym.make('SpaceInvaders-v0')
    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
    print(episode_rewards)


if __name__ == '__main__':
    tf.app.run()