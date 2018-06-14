"""TensorBoard"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def make_session(num_cpu=6):
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    sess_config.gpu_options.allow_growth = True
    return tf.Session(config=sess_config)


def initialize_variables(sess: tf.Session):
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
    ])


def main(_):
    x = tf.placeholder(tf.float32, (None, 100))
    pred = tf.layers.dense(x, 1)
    y = tf.placeholder(tf.float32, (None, 1))
    losses = tf.losses.mean_squared_error(y, pred)
    loss = tf.reduce_mean(losses)
    summary_op = tf.summary.merge([
        tf.summary.histogram('losses', losses),
        tf.summary.scalar('loss', loss)
    ])
    
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    sess = make_session()
    initialize_variables(sess)
    
    summaries_dir = './logdir'
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')
    
    # Record train set summaries, and train
    for n in range(1000):
        x_train = np.random.uniform(size=(32, 100))
        y_train = np.random.uniform(size=(32, 1))
        
        summary, _ = sess.run([summary_op, train_op], feed_dict={x: x_train, y: y_train})
        train_writer.add_summary(summary, n)
        if n % 100 == 0:  # Record summaries and test-set accuracy
            x_test = np.random.uniform(size=(64, 100))
            y_test = np.random.uniform(size=(64, 1))
            summary = sess.run(summary_op, feed_dict={x: x_test, y: y_test})
            test_writer.add_summary(summary, n)
            print('Accuracy at step %s' % (n))


if __name__ == '__main__':
    tf.app.run()