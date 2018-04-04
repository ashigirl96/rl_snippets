"""Experiment whether the eigen image have"""

import gym
import numpy as np
import tensorflow as tf


def main(_):
  env = gym.make('Pong-v0')
  
  observs = [env.reset()]
  
  for _ in range(100):
    observs.append(env.step(env.action_space.sample())[0])
  
  observs = np.stack(observs)
  observs = observs.astype(np.float32)
  
  observs = tf.image.resize_images(observs, [80, 80])
  observs = tf.reduce_mean(observs, 3)
  # observs = tf.reduce_mean(observs, 0)
  print(observs.shape)
  eig_observs = tf.self_adjoint_eig(observs)
  print(eig_observs)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run()