"""
TensorFlow Scan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def main(_):
    p = []
    def fn(previous_output, current_input):
        print(current_input.get_shape())
        p.append(previous_output + current_input)
        return previous_output + current_input
    
    # x = tf.constant([[1, 2, 3], [4, 5, 6]])
    x = tf.ones((4, 32))
    # initializer = tf.constant([0, 0, 0])
    initializer = tf.zeros((32,))
    
    y = tf.scan(fn, x, initializer=initializer)
    
    with tf.Session() as sess:
        print(sess.run(y))
        print(p)


if __name__ == '__main__':
    tf.app.run()