"Experimental gather action"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.enable_eager_execution()

model = tf.layers.Dense(4)

batch_size = 10
x = tf.random_normal((batch_size, 5))
action = 3
y = model(x)
action_prob = tf.nn.softmax(y)
picked_action_prob = tf.gather(action_prob, action, axis=1)
print(picked_action_prob)


