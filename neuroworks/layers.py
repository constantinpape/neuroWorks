# wrapping tensorflow functionality to define relevant layers

from __future__ import division # we want python 3 division syntax
import tensorflow as tf


# TODO
def conv_2d_Layer():
    """
    Conv 2d layer without dropout
    @params:
    """

# TODO understand this
def pixel_wise_softmax_2(output):
    exponential_map = tf.exp(output)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(output)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)
