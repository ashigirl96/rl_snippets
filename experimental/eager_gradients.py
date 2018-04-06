"""Experimental eager gradient & stop gradient"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

tf.enable_eager_execution()


def loss_fn(x, y, model):
    return model(x)


grad_fn = tfe.implicit_gradients(loss_fn)
val_grad_fn = tfe.implicit_value_and_gradients(loss_fn)

model = tf.layers.Dense(10, use_bias=False)
x = tf.random_normal((3, 2))
y = tf.random_normal((3, 2))

value, grads_and_vars = val_grad_fn(x, y, model)
_grads_and_vars = grad_fn(x, y, model)
print('Value of loss: {0}\n\n'.format(value))
np.testing.assert_equal(loss_fn(x, y, model).numpy(), value.numpy())

optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer.apply_gradients(grads_and_vars)

for grad, var in grads_and_vars:
    print("Grad: {0}\nVar: {1}\n\n".format(grad.shape, var))


