from tensorflow.python.client import device_lib
import tensorflow as tf


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

tf.enable_eager_execution()

with tf.device('/gpu:0'):
    a = tf.random_normal((10, 2))
