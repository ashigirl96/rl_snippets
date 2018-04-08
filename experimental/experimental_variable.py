from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import numpy as np
import tensorflow as tf
from ray.experimental import TensorFlowVariables


class Model(object):
    
    def __init__(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        self._build_model()
        self._set_loss()
        optimizer = tf.train.GradientDescentOptimizer(1e-5)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        
        self.variables = TensorFlowVariables(self.loss, self.sess)
        
        self.sess.run(tf.global_variables_initializer())
    
    def _build_model(self):
        self.inputs = tf.placeholder(tf.float32, (None, 7))
        self.labels = tf.placeholder(tf.float32, (None, 2))
        x = tf.layers.dense(self.inputs, 5, use_bias=False)
        x = tf.layers.dense(x, 3, use_bias=False)
        x = tf.layers.dense(x, 2, use_bias=False)
        self.model = x
    
    def _set_loss(self):
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.labels, logits=self.model)
        self.loss = tf.reduce_mean(losses)
    
    def train(self, x, y):
        _, loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={self.inputs: x, self.labels: y})
        return loss


def main(_):
    model = Model()
    pprint(model.variables.get_weights())
    
    x = np.random.uniform(size=(100, 7))
    y = np.random.uniform(size=(100, 2))
    
    print('loss: {0}'.format(model.train(x, y)))


if __name__ == '__main__':
    tf.app.run()