import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


def loss_fn(x, y, model):
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=model(x))


class Model(object):
    
    def __init__(self):
        """Define the layers used, during initialization."""
        self.dense = tf.layers.Dense(10, use_bias=False)
        self.output_layer = tf.layers.Dense(10, use_bias=False)
    
    def __call__(self, observ):
        hidden = self.dense(observ)
        logits = self.output_layer(hidden)
        return logits


def main(_):
    val_grad_fn = tfe.implicit_value_and_gradients(loss_fn)
    model = Model()
    x = tf.random_normal((3, 2))
    y = tf.random_normal((3, 10))
    value, grads_and_vars = val_grad_fn(x, y, model)
    
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer.apply_gradients(grads_and_vars)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.app.run()