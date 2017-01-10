# wrapping tensorflow functionality to define relevant layers

from __future__ import division # we want python 3 division syntax
import tensorflow as tf

#
# Weights and Biases
#

def weight_variable(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

#
# Convolutional Layers
#

def conv2d(x, w, strides = [1,1,1,1], padding = 'SAME'):
    """
    Conv 2d layer without dropout
    """
    return tf.nn.conv2d(x, w, strides=strides, padding=padding)


def conv2d_dropout(x, w, keep_prob, strides = [1,1,1,1], padding = 'SAME'):
    """
    Conv 2d layer with dropout
    """
    l = tf.nn.conv2d(x, w, strides=strides, padding=padding)
    return tf.nn.dropout(l, keep_prob)


# TODO implement residual bottleneck layers
def residual_bottleneck2d():
    pass


def residual_bottleneck2d_dropout():
    pass

#
# Up / Downnscaling Layers
#

def deconv2d(x, w, stride, padding = 'SAME'):
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding=padding)


def max_pool2d(x, pool, stride = None, padding = 'SAME'):
    if stride == None:
        stride = pool
    return tf.nn.max_pool(x, ksize=[1, pool, pool, 1], strides=[1, stride, stride, 1], padding=padding)


# TODO implement
def residual_bottleneck_deconv2d():
    pass

#
# Concatenation
#

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat(3, [x1_crop, x2])

#
# Output Layers
#

# TODO understand this
def pixel_wise_softmax_2(output):
    exponential_map = tf.exp(output)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(output)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)
