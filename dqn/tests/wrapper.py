import gym
import tensorflow as tf
from ray.rllib.utils.atari_wrappers import wrap_deepmind


def main(_):
    env = gym.make('SpaceInvaders-v0')
    env = wrap_deepmind(env)
    observ = env.reset()
    print(observ.shape)
    action = env.action_space.sample()
    print(env.step(action)[1])
    print(env.step(action)[1])
    print(env.step(action)[1])


if __name__ == '__main__':
    tf.app.run()