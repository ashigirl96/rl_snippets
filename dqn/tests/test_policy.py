"""Test Policy"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


def main(_):
    eps = 0.75
    batch_size = 32
    action_size = 4
    x = tf.random_uniform([batch_size, action_size], maxval=0.5)
    y = tf.ones([32, 1])
    logits = tf.concat([x, y], axis=1)
    argmax_action = tf.argmax(logits, axis=1)
    
    def eps_greedy():
        shape = logits.shape.as_list()
        policy = np.ones(shape) * eps / action_size
        print(policy)
        _argmax_action = np.asarray(argmax_action)
        policy[_argmax_action] += 1 - eps
        return policy
    
    policy = tf.py_func(eps_greedy, [], [tf.float32])
    print(policy)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        print(sess.run(policy))


if __name__ == '__main__':
    # tfe.enable_eager_execution()
    tf.app.run()