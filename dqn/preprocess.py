"""Pre process the observation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ATARI(object):
    def __init__(self):
        self.observ_ = tf.placeholder(tf.uint8, [210, 160, 3])
        self.processed_observ = self._atari_preprocess(self.observ_)
    
    def _atari_preprocess(self, observ: tf.Tensor):
        """Process a raw Atari images. Resize it and converts it to gray scale.
        
        Args:
            observ: Placeholder as observation.

        Returns:
            processed observation.
        """
        _size = 84, 84
        with tf.name_scope('atari_preprocess'):
            x = tf.image.rgb_to_grayscale(observ)
            x = tf.image.crop_to_bounding_box(x,
                                              offset_height=34,
                                              offset_width=0,
                                              target_height=160,
                                              target_width=160)
            x = tf.image.resize_images(x, size=_size,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.squeeze(x)
            x.set_shape(_size)
            return x
    
    def __call__(self, observ, sess: tf.Session):
        return sess.run(self.processed_observ, feed_dict={self.observ_: observ})


atari_preprocess = ATARI()
